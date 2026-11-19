# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole B32 hidden-residual CCL families with pooled attention and packed I/O."""
import argparse
import json
import sys
from pathlib import Path
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fusion_reference import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_optimization import FamilyCandidate

class BatchedFamilyCandidate(MultichipDecoder):
    packed_hidden = True
    persistent = False
    fused_gather_enabled = False
    fused_scatter_enabled = False
    communication_dtype = ttnn.bfloat16
    norm_communication_dtype = ttnn.bfloat16
    batch_size = 32

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        kwargs.update(native_local=False, residual_layout="sharded", collective="separate")
        self = super().from_state_dict(*args, **kwargs)
        self.prefill_headroom = ttnn.get_memory_view(self.device, ttnn.BufferType.L1).total_bytes_per_bank - 64*1024
        self.rs_index = 0
        self.persistent_rs, self.persistent_ag = {}, {}
        if self.persistent:
            n=self.batch_size
            def empty(shape):
                return ttnn.empty(shape,dtype=self.communication_dtype,layout=ttnn.TILE_LAYOUT,
                                  device=self.device,memory_config=ttnn.L1_MEMORY_CONFIG)
            self.persistent_rs[n]=[[empty((2,1,32,4096)),empty((1,1,n,1024))] for _ in range(2)]
        return self

    family_reduce = FamilyCandidate.reduce
    fused_gather = FamilyCandidate.fused_gather

    def family_norm(self, x, name):
        x=ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG)
        stats=self.gather(ttnn.rms_norm_pre_all_gather(x,compute_kernel_config=self.compute,dtype=ttnn.bfloat16))
        out=ttnn.rms_norm_post_all_gather(x,stats,epsilon=self.config.rms_norm_eps,
             weight=self.weights[name+".weight"],compute_kernel_config=self.compute,memory_config=ttnn.L1_MEMORY_CONFIG)
        out=ttnn.typecast(out,self.norm_communication_dtype)
        return out if self.fused_gather_enabled else self.gather(out)

    def family_linear(self, x, name):
        if self.fused_gather_enabled and name=="packed_input":
            return self.fused_gather(x,name)
        if self.fused_scatter_enabled and name in ("linear_attn.out_proj","self_attn.o_proj","mlp.down_proj"):
            return FamilyCandidate.linear(self,x,name)
        im,om,cfg=self.dram_layouts[name]
        result=ttnn.linear(ttnn.to_memory_config(x,im),self.dram_weights[name],program_config=cfg,
                 compute_kernel_config=self.matmul_compute(name),memory_config=om,dtype=ttnn.bfloat16)
        return ttnn.to_memory_config(result,ttnn.L1_MEMORY_CONFIG)

    def family_finish(self, x, mixed):
        residual=ttnn.add(x,mixed)
        h=self.family_norm(residual,"post_attention_layernorm")
        if self.fused_gather_enabled:
            h=self.fused_gather(h,"packed_mlp",True)
        else:
            gate=self.family_linear(h,"mlp.gate_proj")
            up=self.family_linear(h,"mlp.up_proj")
            h=ttnn.multiply(gate,up,input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual,self.family_reduce(self.family_linear(h,"mlp.down_proj")))

    def decode_batch_packed(self, x, *, states, current_positions, page_tables):
        """TP4 BF16 [1,1,B,4096], 1<=B<=32, local L1 residual across layers.

        Request attention/cache state stays independent. Dense projections, norms,
        residual adds and reductions share the physical tile. The caller packs at
        stack entry and unpacks only at its exit, outside decoder layer boundaries.
        """
        n = x.shape[2]
        assert self.residual_layout == "sharded" and 1 <= n <= 32
        assert n == len(states) == len(current_positions) == len(page_tables)
        regions = [
            (
                (id(st.conv), id(st.recurrent), st.state_row)
                if self.kind == "linear_attention"
                else (id(st.key), id(st.value), st.page_offset)
            )
            for st in states
        ]
        assert len(set(regions)) == n, "A decode batch must contain distinct request state regions"
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        packed = self.family_linear(self.family_norm(x, "input_layernorm"), "packed_input")
        pooled_linear = (
            self.kind == "linear_attention"
            and all(
                st.conv is states[0].conv
                and st.recurrent is states[0].recurrent
                and getattr(st, "state_row", None) == i
                for i, st in enumerate(states)
            )
            and states[0].conv.shape[0] == n
        )
        independent_linear = all(getattr(st, "state_row", None) is None for st in states)
        if n > 1 and (self.kind == "full_attention" or pooled_linear or independent_linear):
            if self.kind == "full_attention":
                local = self.full_attention_decode_batch(packed, states, current_positions, page_tables)
            elif pooled_linear:
                local = self.pooled_linear_attention_decode_batch(packed, states)
            else:
                local = self.linear_attention_decode_batch(packed, states)
        else:
            local = []
            for i, (state, position, table) in enumerate(zip(states, current_positions, page_tables)):
                row = packed[:, :, i : i + 1, :]
                local.append(
                    self.linear_attention(row, state, decode=True, packed=row, project=False)
                    if self.kind == "linear_attention"
                    else self.full_attention(
                        row, state, page_table=table, position=position, decode=True, packed=row, project=False
                    )
                )
            local = ttnn.concat(local, dim=2) if n > 1 else local[0]
        name = "linear_attn.out_proj" if self.kind == "linear_attention" else "self_attn.o_proj"
        return self.family_finish(x, self.family_reduce(self.family_linear(local, name)))

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("--family",required=True,choices=["separate","persistent","bf8","bf8_persistent","gather","scatter","fused","mixed_fused"])
    args,rest=ap.parse_known_args()
    policy=args.family
    BatchedFamilyCandidate.batch_size=int(rest[rest.index("--batch")+1])
    BatchedFamilyCandidate.persistent="persistent" in policy
    BatchedFamilyCandidate.fused_gather_enabled=policy in ("gather","fused","mixed_fused")
    BatchedFamilyCandidate.fused_scatter_enabled=policy in ("scatter","fused","mixed_fused")
    if "bf8" in policy or policy=="mixed_fused":BatchedFamilyCandidate.communication_dtype=ttnn.bfloat8_b
    if "bf8" in policy:BatchedFamilyCandidate.norm_communication_dtype=ttnn.bfloat8_b
    from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_packed_stack as stack
    stack.MultichipDecoder=BatchedFamilyCandidate
    sys.argv=[sys.argv[0],*rest];stack.main()
    output=Path(rest[rest.index("--output")+1]);data=json.loads(output.read_text());data["batched_family"]=policy
    output.write_text(json.dumps(data,indent=2)+"\n")

if __name__=="__main__":main()

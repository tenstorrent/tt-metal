# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""H2 diagnostic: fixed-shape linear prefill with runtime valid-row state.

This is an experiment, not a production fallback. The caller allocates every
profile before capture, warms ``forward`` for every profile/state binding, and
restores request state before admitting real work. ``refresh`` is a request-time
host copy into preallocated buffers; ``forward`` contains only device work.
Inputs are BF16 residuals [1,1,bucket,hidden_width] in the decoder's existing
residual layout. Dummy rows can contain arbitrary finite values. Returned rows
are physical; only the caller's valid prefix is meaningful.

Host inputs per chunk: valid_count; residual rows padded to select_bucket(count).
Absolute position is irrelevant to linear attention but the caller must supply
the state from precisely that absolute position. History indices are
[valid_count, valid_count+1, valid_count+2] into [old_history; projected_rows].
They are expanded across all 2048 channels because gather does not broadcast,
then stored in [1,1,2048,3] orientation to gather on the last dimension. This
avoids gather's output-transform allocation and keeps its ``out`` binding fixed.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import DecoderState, MultichipDecoder

LINEAR_BUCKETS = (1, 32, 64, 128, 511, 512, 640, 768, 960, 1024, 4096, 8192, 32768)


def select_bucket(valid_count):
    if not 1 <= valid_count <= 32768:
        raise ValueError("Split linear chunks at the original 32768 cap before bucketing")
    return next(n for n in LINEAR_BUCKETS if n >= valid_count)


class NormalizedLinearPrefill:
    """Wrap an existing real-weight decoder without changing its policy knobs."""

    def __init__(self, decoder, *, buckets=LINEAR_BUCKETS):
        if decoder.kind != "linear_attention":
            raise ValueError("This diagnostic only adapts linear attention")
        self.decoder = decoder
        self.device = decoder.device
        self.profiles = {}
        for n in buckets:
            if n not in LINEAR_BUCKETS or n > decoder.chunk_size:
                raise ValueError(f"Unreviewed physical profile {n}")
            self.profiles[n] = {
                "g_mask": self.upload(torch.ones(1, n, 8), dtype=ttnn.float32),
                "beta_mask": self.upload(torch.ones(1, n, 8)),
                "history_indices": self.upload(
                    self.history_indices(n), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                "history_output": self.upload(torch.zeros(1, 1, 2048, 3), layout=ttnn.ROW_MAJOR_LAYOUT),
            }

    @staticmethod
    def history_indices(valid_count):
        return (
            torch.arange(valid_count, valid_count + 3, dtype=torch.int64)
            .reshape(1, 1, 1, 3)
            .expand(1, 1, 2048, 3)
            .contiguous()
        )

    def upload(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=True):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device if device else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def refresh(self, valid_count):
        """Outside capture: copy masks and indices; never allocate device storage."""
        n = select_bucket(valid_count)
        profile = self.profiles[n]
        mask = (torch.arange(n) < valid_count).float().reshape(1, n, 1).expand(1, n, 8).contiguous()
        for name, value, dtype, layout in (
            ("g_mask", mask, ttnn.float32, ttnn.TILE_LAYOUT),
            ("beta_mask", mask, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("history_indices", self.history_indices(valid_count), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ):
            ttnn.copy_host_to_device_tensor(self.upload(value, dtype=dtype, layout=layout, device=False), profile[name])
        return n

    def attention(self, x, state, *, profile):
        d = self.decoder
        if getattr(state, "state_row", None) is not None:
            local = DecoderState(
                conv=d.read_linear_state(state, "conv"), recurrent=d.read_linear_state(state, "recurrent")
            )
            out = self.attention(x, local, profile=profile)
            d.write_linear_state(state, "conv", local.conv)
            d.write_linear_state(state, "recurrent", local.recurrent)
            return out

        n = x.shape[2]
        packed = d.linear(x, "packed_input")
        z = packed[:, :, :, 2048:3072]
        beta = ttnn.reshape(ttnn.sigmoid(packed[:, :, :, 3072:3080]), (1, n, 8))
        a = ttnn.typecast(ttnn.reshape(packed[:, :, :, 3104:3112], (1, n, 8)), ttnn.float32)
        g = ttnn.multiply(
            ttnn.add(a, d.dt_bias),
            d.decay,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
        )
        # Preserve the original FP32 g / BF16 beta policy. A finite inactive
        # projected row contributes exactly exp(0)=1 and beta=0 to recurrence.
        g = ttnn.multiply(g, profile["g_mask"])
        beta = ttnn.multiply(beta, profile["beta_mask"])
        old_history = ttnn.reshape(ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT), (1, 3, 2048))
        padded = packed
        if n % 32:
            padded = ttnn.pad(packed, [(0, 0), (0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            ttnn.reshape(padded, (1, padded.shape[2], 3136)),
            old_history,
            *d.taps,
            512,
            512,
            1024,
            actual_start=d.conv_start,
            predecessor_carry=old_history,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
            compute_kernel_config=d.conv_compute,
        )
        # All tensor shapes/slice offsets below depend on the physical profile,
        # never valid_count. BF16 + RM matches gather's supported value/index
        # layouts; full-channel indices avoid relying on nonexistent broadcast.
        projected = ttnn.to_layout(packed[:, :, :, :2048], ttnn.ROW_MAJOR_LAYOUT)
        history_source = ttnn.concat([ttnn.reshape(old_history, (1, 1, 3, 2048)), projected], dim=2)
        history_source = ttnn.transpose(history_source, 2, 3)
        history = ttnn.gather(history_source, 3, profile["history_indices"], out=profile["history_output"])
        ttnn.copy(ttnn.transpose(history, 2, 3), state.conv)

        if n % 32:
            g = ttnn.pad(g, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
            beta = ttnn.pad(beta, [(0, 0), (0, 32 - n % 32), (0, 0)], 0.0)
        out, new_state = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state.recurrent,
            output_final_state=True,
            compute_kernel_config=d.compute,
            chunk_size=32,
            num_key_heads=4,
            output_head_major=True,
            **d.gdn_constants,
        )
        ttnn.copy(new_state, state.recurrent)
        out = d.norm(ttnn.typecast(out, ttnn.bfloat16), "linear_attn.norm")
        out = ttnn.transformer.concatenate_heads(ttnn.reshape(out, (1, 8, padded.shape[2], 128)))
        out = ttnn.reshape(out, (1, 1, padded.shape[2], 1024))[:, :, :n, :]
        out = ttnn.multiply(z, out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return d.reduce(d.linear(out, "linear_attn.out_proj"))

    def forward(self, residual, *, state):
        """Device-only one-bucket forward. Call refresh before request admission."""
        n = residual.shape[2]
        if residual.shape[-1] != self.decoder.hidden_width:
            raise ValueError("Preserve the decoder's inter-layer residual layout")
        profile = self.profiles[n]
        h = self.decoder.norm(residual, "input_layernorm")
        mixed = self.attention(h, state, profile=profile)
        return ttnn.to_memory_config(self.decoder.finish(residual, mixed), ttnn.DRAM_MEMORY_CONFIG)


def main():
    """Small real-weight numerical probe; run only under the parent's timeout."""
    from transformers import AutoConfig

    from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import Checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1,2,3,31,33,34,65,129,481,511,512,513")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-check", action="store_true")
    args = parser.parse_args()
    lengths = [int(n) for n in args.lengths.split(",")]
    torch.set_num_threads(8)
    torch.manual_seed(834)
    checkpoint = Checkpoint()
    config = AutoConfig.from_pretrained(checkpoint.path).text_config
    weights = checkpoint.tensors("model.language_model.layers.0.")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    records = []
    try:
        decoder = MultichipDecoder.from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh)
        buckets = {select_bucket(n) for n in lengths}
        if args.cache_check:
            buckets.add(64)
        adapter = NormalizedLinearPrefill(decoder, buckets=sorted(buckets))

        def host_shards(tensor):
            return [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(tensor)]

        def compare(expected, actual):
            pairs = list(zip(host_shards(expected), host_shards(actual)))
            scores = []
            errors = []
            equals = []
            for a, b in pairs:
                b = b[:, :, : a.shape[2], :] if b.shape != a.shape else b
                equals.append(torch.equal(a, b))
                scores.append(
                    1.0 if equals[-1] else torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1].item()
                )
                errors.append((a - b).abs().max().item())
            return {"pcc": scores, "max_abs": errors, "bit_equal": equals}

        for valid in lengths:
            n = adapter.refresh(valid)
            reference_state = decoder.allocate_state(context=32)
            normalized_state = decoder.allocate_state(context=32)
            # Seed a nonzero three-token prefix so valid_count 1/2 cannot pass
            # by accidentally substituting zeros for retained history rows.
            prefix = adapter.upload(torch.randn(1, 1, 3, decoder.hidden_width, dtype=torch.bfloat16))
            decoder.prefill_forward(prefix, state=reference_state)
            ttnn.copy(reference_state.conv, normalized_state.conv)
            ttnn.copy(reference_state.recurrent, normalized_state.recurrent)
            real = torch.randn(1, 1, valid, decoder.hidden_width, dtype=torch.bfloat16)
            physical = torch.randn(1, 1, n, decoder.hidden_width, dtype=torch.bfloat16) * 7
            physical[:, :, :valid] = real
            ref = decoder.prefill_forward(adapter.upload(real), state=reference_state, start_position=3)
            result = adapter.forward(adapter.upload(physical), state=normalized_state)
            record = {
                "valid": valid,
                "bucket": n,
                "output": compare(ref, result),
                "conv": compare(reference_state.conv, normalized_state.conv),
                "recurrent": compare(reference_state.recurrent, normalized_state.recurrent),
            }
            # A following real decode detects dummy-row damage hidden by
            # apparently correct real prefill outputs.
            next_row = adapter.upload(torch.randn(1, 1, 1, decoder.hidden_width, dtype=torch.bfloat16))
            position = adapter.upload(
                torch.tensor([3 + valid], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ref_next = decoder.decode_forward(next_row, state=reference_state, current_position=position)
            got_next = decoder.decode_forward(next_row, state=normalized_state, current_position=position)
            record["decode_after"] = compare(ref_next, got_next)
            records.append(record)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            # Existing decoder probes use 0.995, not a newly relaxed bar.
            assert all(
                v >= 0.995 for key in ("output", "conv", "recurrent", "decode_after") for v in record[key]["pcc"]
            )
        if args.cache_check:
            # All persistent device inputs/state are created before preparation.
            # Repeated calls vary only runtime mask/index contents. No result is
            # retained across forwards, and no host output conversion participates
            # in the measurement (its helper programs would obscure attribution).
            cache_state = decoder.allocate_state(context=32)
            cache_input = adapter.upload(torch.randn(1, 1, 64, decoder.hidden_width, dtype=torch.bfloat16))
            for valid in (33, 34, 33, 34):
                adapter.refresh(valid)
                cache_output = adapter.forward(cache_input, state=cache_state)
                del cache_output
            ttnn.synchronize_device(mesh)
            before = mesh.num_program_cache_entries()
            observations = []
            for valid in (34, 33, 34, 33, 33, 34):
                adapter.refresh(valid)
                cache_output = adapter.forward(cache_input, state=cache_state)
                del cache_output
                ttnn.synchronize_device(mesh)
                observations.append({"valid": valid, "program_cache_entries": mesh.num_program_cache_entries()})
            cache_record = {
                "bucket": 64,
                "warm_valid_counts": [33, 34, 33, 34],
                "before": before,
                "observations": observations,
                "after": mesh.num_program_cache_entries(),
                "passed": all(item["program_cache_entries"] == before for item in observations),
            }
            cache_path = Path(args.output).with_suffix(".cache.json")
            cache_path.write_text(json.dumps(cache_record, indent=2) + "\n")
            print("PROGRAM_CACHE_CHECK " + json.dumps(cache_record), flush=True)
            assert cache_record["passed"], f"Runtime valid_count changed the program cache: {cache_path}"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

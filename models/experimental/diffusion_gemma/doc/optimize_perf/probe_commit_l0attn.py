# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight probe: compare layer-0 ATTENTION OUTPUT of the batched vs sequential commit.

The KV-cache verify shows layer-0 K/V bit-exact but layer-1 K/V diverging (~0.94),
i.e. layer-0's OUTPUT hidden state diverges. This probe isolates whether that comes
from the attention (captures layer-0 attn output from both paths) or downstream MLP.
Builds a 2-layer model, prefills, captures one denoise block's committed tokens, then
runs the sequential and batched commits with the layer-0 attention output captured.
"""
from __future__ import annotations

import os
import sys

import torch
from loguru import logger


def _pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel():
        return -1.0
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def main():
    import ttnn

    ckpt = os.environ["DG_CKPT"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        from models.experimental.diffusion_gemma.checkpoint import (
            build_tt_model_from_checkpoint_dir,
            text_generation_prefixes_for_layers,
        )
        from models.experimental.diffusion_gemma.config import DiffusionConfig
        from models.experimental.diffusion_gemma.tt import commit_batched, commit_decode
        from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block, tokenize_prompt
        from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession

        mi = build_tt_model_from_checkpoint_dir(
            mesh, ckpt, state_prefixes=text_generation_prefixes_for_layers(2), num_layers=2, max_seq_len=512
        )
        tt_model = mi.tt_model
        cfg = DiffusionConfig(canvas_length=256)
        prompt = tokenize_prompt(mi.tokenizer, "The capital of France is")
        session = BlockDiffusionServingSession(
            tt_model, mi.state_dict, config=cfg, tokenizer=mi.tokenizer, gumbel_mode="argmax", seed=0
        )
        session.prefill(prompt)
        start_pos = session.next_pos

        captured = {}
        # Wrap layer-0 shared_mlp to capture INPUT (normed) and OUTPUT for both paths.
        l0 = tt_model.layers[0]
        orig_mlp = l0.shared_mlp
        mlp_cap = {"seq_in": [], "seq_out": [], "bat_in": None, "bat_out": None}

        def mlp_wrap(x, *a, **kw):
            xt = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float().clone()
            out = orig_mlp(x, *a, **kw)
            ot = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone()
            if mlp_cap.get("_mode") == "seq":
                mlp_cap["seq_in"].append(xt)
                mlp_cap["seq_out"].append(ot)
            elif mlp_cap.get("_mode") == "bat":
                mlp_cap["bat_in"] = xt
                mlp_cap["bat_out"] = ot
            return out

        l0.shared_mlp = mlp_wrap

        # Wrap layer-0 MoE forward (router+experts) to capture output for both paths.
        moe_cap = {"seq": [], "bat": None, "_mode": None}
        orig_bmoe = commit_batched._denoise_moe_forward
        orig_smoe = commit_decode._commit_moe_forward

        def bmoe_wrap(moe, router_input, expert_input, *a, **kw):
            out = orig_bmoe(moe, router_input, expert_input, *a, **kw)
            if moe is l0.moe:
                moe_cap["bat"] = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone()
            return out

        def smoe_wrap(moe, *a, **kw):
            out = orig_smoe(moe, *a, **kw)
            if moe is l0.moe:
                moe_cap["seq"].append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone())
            return out

        # Capture layer-0 router output (dense_routing) for both paths.
        from models.experimental.diffusion_gemma.tt import denoise_forward as _dfwd

        rt_cap = {"seq": [], "bat": None}
        orig_brt = _dfwd._denoise_router_forward
        orig_srt = commit_decode._commit_router_forward
        l0_router = l0.moe.router

        def brt_wrap(router, hs, *a, **kw):
            out = orig_brt(router, hs, *a, **kw)
            if router is l0_router:
                rt_cap["bat"] = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone()
            return out

        def srt_wrap(router, hs, *a, **kw):
            out = orig_srt(router, hs, *a, **kw)
            if router is l0_router:
                rt_cap["seq"].append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone())
            return out

        def _noop(_m, canvas_tokens, **_k):
            captured["committed"] = canvas_tokens.clone()

        denoise_and_commit_block(
            tt_model,
            session._logits_fn,
            session._init_canvas_fn(0, start_pos),
            cfg,
            start_pos=start_pos,
            gumbel_noise_fn=session._gumbel_noise_fn(0),
            noise_tokens_fn=session._noise_tokens_fn(0),
            commit_fn=_noop,
        )
        committed = captured["committed"]
        l0_attn = tt_model.layers[0].self_attn

        # --- capture BATCHED layer-0 attn output ---
        batched_cap = {}
        orig_b = commit_batched._commit_attention_batched

        def wrap_b(attn, canvas_hidden, **kw):
            out = orig_b(attn, canvas_hidden, **kw)
            if attn is l0_attn and "batched" not in batched_cap:
                batched_cap["batched"] = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone()
            return out

        # --- capture SEQUENTIAL layer-0 attn output (per token, concat) ---
        seq_chunks = []
        orig_s = commit_decode._commit_attention_decode_forward

        def wrap_s(*a, **kw):
            out = orig_s(*a, **kw)
            weights = kw.get("weights", a[3] if len(a) > 3 else None)
            if weights is l0_attn.weights:
                seq_chunks.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().clone())
            return out

        # sequential into the model caches (mutates), batched into clones
        def clone_caches(kvc):
            by = {}
            cl = []
            for kv in kvc:
                k = id(kv[0])
                if k not in by:
                    by[k] = [ttnn.clone(kv[0]), ttnn.clone(kv[1])]
                cl.append(by[k])
            return cl

        orig_caches = tt_model.tt_kv_cache
        clones = clone_caches(orig_caches)

        commit_decode._commit_attention_decode_forward = wrap_s
        commit_decode._commit_moe_forward = smoe_wrap
        commit_decode._commit_router_forward = srt_wrap
        mlp_cap["_mode"] = "seq"
        try:
            from models.experimental.diffusion_gemma.tt.generate import commit_canvas_tokens

            commit_canvas_tokens(tt_model, committed, start_pos=start_pos, page_table=None)
        finally:
            commit_decode._commit_attention_decode_forward = orig_s
            commit_decode._commit_moe_forward = orig_smoe
            commit_decode._commit_router_forward = orig_srt

        tt_model.tt_kv_cache = clones
        commit_batched._commit_attention_batched = wrap_b
        commit_batched._denoise_moe_forward = bmoe_wrap
        _dfwd._denoise_router_forward = brt_wrap
        mlp_cap["_mode"] = "bat"
        try:
            commit_batched.commit_canvas_tokens_batched(tt_model, committed, start_pos=start_pos, page_table=None)
        finally:
            commit_batched._commit_attention_batched = orig_b
            commit_batched._denoise_moe_forward = orig_bmoe
            _dfwd._denoise_router_forward = orig_brt
            tt_model.tt_kv_cache = orig_caches
            l0.shared_mlp = orig_mlp

        if rt_cap["seq"] and rt_cap["bat"] is not None:
            sr = torch.cat(rt_cap["seq"], dim=-2).reshape(-1, rt_cap["bat"].shape[-1])
            br = rt_cap["bat"].reshape(-1, rt_cap["bat"].shape[-1])
            n = min(sr.shape[0], br.shape[0])
            print(f"L0-ROUTER      PCC={_pcc(sr[:n], br[:n]):.6f}  max={float((sr[:n]-br[:n]).abs().max()):.3e}")
            # nonzero-pattern (which experts selected) agreement
            snz = sr[:n].abs() > 1e-6
            bnz = br[:n].abs() > 1e-6
            agree = (snz == bnz).float().mean().item()
            print(
                f"L0-ROUTER  selected-expert-mask agreement={agree:.4f}  "
                f"seq_nnz/tok={snz.sum(-1).float().mean():.2f} bat_nnz/tok={bnz.sum(-1).float().mean():.2f}"
            )

        if moe_cap["seq"] and moe_cap["bat"] is not None:
            sm = torch.cat(moe_cap["seq"], dim=-2).reshape(-1, moe_cap["bat"].shape[-1])
            bm = moe_cap["bat"].reshape(-1, moe_cap["bat"].shape[-1])
            n = min(sm.shape[0], bm.shape[0])
            print(f"L0-MOE-OUTPUT  PCC={_pcc(sm[:n], bm[:n]):.6f}  max={float((sm[:n]-bm[:n]).abs().max()):.3e}")

        # shared_mlp isolation for layer 0
        if mlp_cap["seq_in"] and mlp_cap["bat_in"] is not None:
            si = torch.cat(mlp_cap["seq_in"], dim=-2).reshape(-1, mlp_cap["bat_in"].shape[-1])
            so = torch.cat(mlp_cap["seq_out"], dim=-2).reshape(-1, mlp_cap["bat_out"].shape[-1])
            bi = mlp_cap["bat_in"].reshape(-1, mlp_cap["bat_in"].shape[-1])
            bo = mlp_cap["bat_out"].reshape(-1, mlp_cap["bat_out"].shape[-1])
            n = min(si.shape[0], bi.shape[0])
            print(f"L0-SHARED_MLP  input  PCC={_pcc(si[:n], bi[:n]):.6f}  max={float((si[:n]-bi[:n]).abs().max()):.3e}")
            print(f"L0-SHARED_MLP  output PCC={_pcc(so[:n], bo[:n]):.6f}  max={float((so[:n]-bo[:n]).abs().max()):.3e}")

        seq_attn = torch.cat(seq_chunks, dim=-2) if seq_chunks else None  # [1,1,256,H]
        bat_attn = batched_cap.get("batched")
        logger.info(
            f"seq_chunks={len(seq_chunks)} seq_attn shape="
            f"{tuple(seq_attn.shape) if seq_attn is not None else None} "
            f"bat_attn shape={tuple(bat_attn.shape) if bat_attn is not None else None}"
        )
        if seq_attn is not None and bat_attn is not None:
            # reshape both to [256, H]
            s = seq_attn.reshape(-1, seq_attn.shape[-1])
            b = bat_attn.reshape(-1, bat_attn.shape[-1])
            n = min(s.shape[0], b.shape[0])
            pcc = _pcc(s[:n], b[:n])
            md = float((s[:n] - b[:n]).abs().max())
            print(f"L0-ATTN-OUTPUT  PCC={pcc:.6f}  max_abs_diff={md:.4e}  rows={n}")
            # per-position first-16 PCC to see if early/late positions differ
            for i in [0, 1, 2, 8, 32, 64, 128, 255]:
                if i < n:
                    print(f"  pos {i:3d}: pcc={_pcc(s[i], b[i]):.5f} max={float((s[i]-b[i]).abs().max()):.3e}")
    finally:
        ttnn.close_mesh_device(mesh)
        if mesh.get_num_devices() > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())

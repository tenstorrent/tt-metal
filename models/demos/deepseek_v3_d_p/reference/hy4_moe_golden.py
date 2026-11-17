# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dump module-level FFN / MoE goldens for Hy4-preview from the fp32 CPU reference on real weights.

The 78-layer golden of #65919 (``reference/hy4/``, PR #65936) stores KV caches, not FFN outputs. The MoE and
dense layer-0 module tests (``tests/pcc/test_hy4_moe.py``, #65890 / #65923) need the FFN input (post-attention
RMSNorm output, after the iHC pre-mix) of a real prompt and what HF ``hy_v4`` makes of it. This runs
``Hy4PrefillReference`` over the first ``--layers`` decoder layers and, per layer, saves:

  * ``hn``          ``[S, hidden]`` fp32: the FFN input, as the TT FFN / MoE module receives it;
  * dense layers:   ``dense_out``  ``[S, hidden]`` (HF ``HYV4MLP``, unclamped);
  * MoE layers:     ``routed_out`` (HF ``HYV4MoE`` routed half), ``shared_out`` (``HYV4MLP``, unclamped),
                    ``topk_idx`` ``[S, k]`` int32 / ``topk_w`` ``[S, k]`` fp32 (router picks and weights, x 2.827),
                    and ``clamp_frac``: the fraction of routed (token, expert) gate / up values the clamp moved.

One ``layer_{i}.safetensors`` per layer under ``--out``. The math is ``reference/hy4/model.py``'s, called rather than
copied (``_moe`` for the routed half; ``reference/hy4_moe.py`` re-derives the picks and the clamp statistics with the
same rule and is checked against it here).

Needs the golden-refs package (``models/demos/deepseek_v3_d_p/reference/hy4``, PR #65936) on the import path. Until
that lands on this branch, run with ``PYTHONPATH`` = a checkout of ``hy4-preview/prefill-golden-refs`` FIRST and this
file by path::

    PYTHONPATH=<golden-refs checkout> python <this file> --model /mnt/tt-data/Hy4-Preview \\
        --golden /mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-v2 --tokens 4096 --layers 2 --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger

F32 = torch.float32


def _load_local(name: str):
    """This branch's ``reference/<name>.py`` by path, so it is found even when PYTHONPATH points at another checkout."""
    spec = importlib.util.spec_from_file_location(f"_hy4_local_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="/mnt/tt-data/Hy4-Preview")
    ap.add_argument("--golden", default="/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-v2")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from safetensors.torch import save_file

    from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import Hy4Checkpoint
    from models.demos.deepseek_v3_d_p.reference.hy4.model import (
        Hy4PrefillReference,
        _blocks,
        hyper_connection,
        rms_norm,
    )

    ref_moe = _load_local("hy4_moe")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    token_ids = json.loads((Path(args.golden) / "metadata.json").read_text())["token_ids"][: args.tokens]

    class _Capture(Hy4PrefillReference):
        """``Hy4PrefillReference`` whose FFN also saves its input and its outputs (same math as ``_ffn``)."""

        def _ffn(self, layer, w, x4):
            cfg = self.cfg
            S = x4.shape[0]
            post = torch.empty(S, cfg.hc_mult, dtype=F32)
            hn = torch.empty(S, cfg.hidden_size, dtype=F32)
            for a, b in _blocks(S, self.proj_block):
                post[a:b], h = hyper_connection(x4[a:b], w["ffn_hc.fn"], w["ffn_hc.base"], w["ffn_hc.scale"], cfg)
                hn[a:b] = rms_norm(h, w["post_attention_layernorm"], cfg.rms_norm_eps)
            rec = {"hn": hn.clone()}
            if cfg.is_dense(layer):
                out = ref_moe.swiglu_mlp(hn, w["mlp.gate_proj"], w["mlp.up_proj"], w["mlp.down_proj"])
                rec["dense_out"] = out.clone()
            else:
                routed = self._moe(w, hn)
                shared = ref_moe.swiglu_mlp(
                    hn, w["mlp.shared.gate_proj"], w["mlp.shared.up_proj"], w["mlp.shared.down_proj"]
                )
                top_i, top_w = ref_moe.route(
                    hn,
                    w["mlp.gate"],
                    w["mlp.gate.bias"],
                    top_k=cfg.num_experts_per_tok,
                    n_group=cfg.n_group,
                    topk_group=cfg.topk_group,
                    norm_topk_prob=cfg.norm_topk_prob,
                    routed_scaling_factor=cfg.routed_scaling_factor,
                )
                gate_frac, up_frac = ref_moe.clamp_fractions(
                    hn, top_i, w["mlp.experts.gate_up_proj"], limit=cfg.swiglu_limit
                )
                # The local re-derivation must agree with model.py's own routed half on these picks.
                check = ref_moe.routed_moe(
                    hn,
                    top_i,
                    top_w,
                    w["mlp.experts.gate_up_proj"],
                    w["mlp.experts.down_proj"],
                    limit=cfg.swiglu_limit,
                )
                err = (check - routed).abs().max().item() / max(routed.abs().max().item(), 1e-30)
                logger.info(f"layer {layer}: local routed reference vs model.py _moe: max rel err {err:.2e}")
                if err > 1e-4:
                    raise AssertionError(f"reference/hy4_moe.py disagrees with model.py _moe on layer {layer}: {err}")
                rec.update(
                    routed_out=routed.clone(),
                    shared_out=shared.clone(),
                    topk_idx=top_i.to(torch.int32),
                    topk_w=top_w.to(F32),
                    clamp_frac=torch.tensor([gate_frac, up_frac], dtype=F32),
                )
                out = routed + shared
            save_file({k: v.contiguous() for k, v in rec.items()}, str(out_dir / f"layer_{layer}.safetensors"))
            logger.info(f"layer {layer}: saved {sorted(rec)} to {out_dir}")
            for a, b in _blocks(S, self.proj_block):
                x4[a:b] += post[a:b].unsqueeze(-1) * out[a:b].unsqueeze(1)

    class _NullSink:
        def on_layer(self, *a, **k):
            pass

    ckpt = Hy4Checkpoint(args.model)
    t0 = time.time()
    _Capture(ckpt).run(torch.tensor(token_ids), _NullSink(), num_layers=args.layers)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "golden": str(args.golden),
                "tokens": len(token_ids),
                "layers": args.layers,
                "precision": ckpt.precision,
                "seconds": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )
    logger.info(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())

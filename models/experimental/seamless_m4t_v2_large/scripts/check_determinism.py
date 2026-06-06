#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run greedy generate twice in one process; compare text tokens and speech metadata."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import ttnn
from transformers import AutoProcessor

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models.experimental.seamless_m4t_v2_large.demo.demo import (  # noqa: E402
    _isolated_task_session,
    _prewarm_vocoder_from_last_generate,
    _process_jit_preflight,
    torch_ids_to_ttnn,
)
from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (  # noqa: E402
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import (  # noqa: E402
    ensure_seamless_m4t_v2_large_weights,
)
from models.experimental.seamless_m4t_v2_large.tt.common import (  # noqa: E402
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)


def _token_list(seq_tt: ttnn.Tensor, *, pad_token_id: int, eos_token_id: int) -> list[int]:
    """Read a ``[1, L]`` greedy-decoder sequence; trim pad and cut at the final EOS."""
    t = to_torch_replicated_first_shard(seq_tt).long().reshape(-1).tolist()
    while t and int(t[-1]) == int(pad_token_id):
        t.pop()
    if int(eos_token_id) in t:
        last_eos = len(t) - 1 - t[::-1].index(int(eos_token_id))
        t = t[: last_eos + 1]
    return [int(x) for x in t]


def _run_iters(
    device,
    tt_model,
    *,
    ids_tt,
    attn_tt,
    gen_common,
    generate_speech: bool,
    tgt_lang: str,
    iters: int,
    label: str,
    pad_token_id: int,
    eos_token_id: int,
    warmup_iters: int = 0,
) -> list[dict]:
    rows: list[dict] = []
    total = warmup_iters + iters
    for i in range(total):
        kw = dict(
            input_ids=ids_tt,
            attention_mask=attn_tt,
            generate_speech=generate_speech,
            tgt_lang=tgt_lang,
            **gen_common,
        )
        kw["return_timings"] = True
        if generate_speech:
            kw["return_intermediate_token_ids"] = True
        out = tt_model.generate(**kw)
        ttnn.synchronize_device(device)
        is_warmup = i < warmup_iters
        if generate_speech:
            toks = _token_list(out.sequences, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
            units = to_torch_replicated_first_shard(out.unit_sequences).long().reshape(-1).tolist()
            samples = int(to_torch_replicated_first_shard(out.waveform_lengths).long().reshape(-1)[0].item())
            if out.timings is not None:
                samples = int(out.timings.output_samples or samples)
            ttnn.deallocate(out.waveform)
            ttnn.deallocate(out.waveform_lengths)
            ttnn.deallocate(out.sequences)
            ttnn.deallocate(out.unit_sequences)
            row = {
                "tokens": toks,
                "units_len": len(units),
                "units_hash": hashlib.md5(str(units[:64]).encode()).hexdigest()[:8],
                "samples": samples,
            }
        else:
            toks = _token_list(out.sequences, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
            ttnn.deallocate(out.sequences)
            row = {"tokens": toks, "units_len": 0, "units_hash": "-", "samples": 0}
        if is_warmup:
            if generate_speech:
                _prewarm_vocoder_from_last_generate(tt_model)
            print(f"  {label} warmup {i + 1}/{warmup_iters}: {len(row['tokens'])} tokens, samples={row['samples']}")
            continue
        rows.append(row)
        print(
            f"  {label} iter {len(rows)}: {len(row['tokens'])} tokens, samples={row['samples']}, units_len={row['units_len']}"
        )
    return rows


def _compare(rows: list[dict], label: str) -> bool:
    base = rows[0]["tokens"]
    ok = True
    for i, r in enumerate(rows[1:], start=2):
        if r["tokens"] != base:
            print(
                f"  DETERMINISM FAIL {label}: iter1 vs iter{i} token mismatch at len {len(base)} vs {len(r['tokens'])}"
            )
            # first diff
            for j, (a, b) in enumerate(zip(base, r["tokens"])):
                if a != b:
                    print(f"    first diff pos {j}: {a} vs {b}")
                    break
            ok = False
        if r["samples"] and r["samples"] != rows[0]["samples"]:
            print(f"  DETERMINISM FAIL {label}: iter1 samples={rows[0]['samples']} iter{i} samples={r['samples']}")
            ok = False
    if ok:
        print(
            f"  DETERMINISM OK {label}: {len(rows)} runs identical ({len(base)} tokens, samples={rows[0]['samples']})"
        )
    return ok


def main() -> None:
    weights_dir = ensure_seamless_m4t_v2_large_weights()
    path = str(weights_dir)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    hf_model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = hf_model.t2u_model.config

    src_text = (
        "going along slushy country roads and speaking to damp audiences in draughty schoolrooms "
        "day after day for a fortnight he'll have to put in an appearance at some place of worship "
        "on sunday morning and he can come to us immediately afterwards"
    )
    text_inputs = processor(text=src_text, src_lang="eng", return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    input_text_attn = text_inputs["attention_mask"]

    gen_common = hf_aligned_generation_kwargs(
        hf_model.generation_config,
        use_kv_cache=True,
        use_decode_trace=True,
        use_2cq=True,
        return_timings=False,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        original_default = None

    session_kw = dict(
        hf_model=hf_model,
        cfg=cfg,
        t2u_cfg=t2u_cfg,
        gen_common=gen_common,
        original_default=original_default,
    )

    _process_jit_preflight(
        session_kw=session_kw,
        input_ids=input_ids,
        input_text_attn=input_text_attn,
        gen_common=gen_common,
    )

    all_ok = True
    with _isolated_task_session(**session_kw) as (device, tt_model):
        ids_tt = torch_ids_to_ttnn(device, input_ids)
        attn_tt = torch_ids_to_ttnn(device, input_text_attn)

        pad_token_id = int(hf_model.generation_config.pad_token_id)
        eos_token_id = int(hf_model.generation_config.eos_token_id)

        print("T2TT x3 (same device, same process):")
        t2tt_rows = _run_iters(
            device,
            tt_model,
            ids_tt=ids_tt,
            attn_tt=attn_tt,
            gen_common=gen_common,
            generate_speech=False,
            tgt_lang="hin",
            iters=3,
            label="T2TT",
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        all_ok &= _compare(t2tt_rows, "T2TT")

        print("T2ST x3 (same device, same process; 1 vocoder warmup):")
        t2st_rows = _run_iters(
            device,
            tt_model,
            ids_tt=ids_tt,
            attn_tt=attn_tt,
            gen_common=gen_common,
            generate_speech=True,
            tgt_lang="hin",
            iters=3,
            label="T2ST",
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            warmup_iters=1,
        )
        all_ok &= _compare(t2st_rows, "T2ST")

    print("")
    print("OVERALL:", "DETERMINISTIC" if all_ok else "NON-DETERMINISTIC")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

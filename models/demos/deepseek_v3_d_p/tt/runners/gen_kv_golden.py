#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""
Offline KV-cache golden generator for prefill_runner.py's PCC validation.

Runs the HuggingFace reference model layer-by-layer on a chosen input
(currently longbook_qa_eng / json_prompts / abc_1k / random) at a given
ISL and num_layers, captures per-layer K/V via DynamicCache, and writes a
`.pt` file at the path
`$TT_DS_PREFILL_HOST_REF_CACHE/{weight_type}_{input_source}_isl{isl}_layers{N}_experts{E}_pad{side}.pt`.

After this script completes, running `prefill_runner.py` with the same
config + `PREFILL_KV_VALIDATE=1` will pick up the golden and run per-layer
PCC.

No MeshDevice required — pure HF + torch on CPU. Memory peak ~21 GB
(load_and_compute_layer_by_layer evicts each layer's weights as it goes).

Required env:
    DEEPSEEK_V3_HF_MODEL=/path/to/DeepSeek-R1-0528
    TT_DS_PREFILL_HOST_REF_CACHE=/path/to/golden/dir   (created if missing)

Knobs (match what prefill_runner.py will use):
    PREFILL_NUM_LAYERS         (default 1)
    PREFILL_MAX_SEQ_LEN        (default 1024)
    PREFILL_KV_GOLDEN_INPUT_SOURCE   (default longbook_qa_eng)
    PREFILL_KV_GOLDEN_PAD_SIDE       (default right)
    PREFILL_KV_GOLDEN_WEIGHT_TYPE    (default pretrained)
"""

import json
import os
from pathlib import Path

from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    INFINITEBENCH_SUBSETS,
    ReferenceCacheKey,
    check_reference_cache_exists,
    download_infinitebench_subset,
    load_and_compute_layer_by_layer,
    save_reference_cache,
    tokenize_prompt_to_isl,
)


def _resolve_prompt_text(input_source: str) -> str:
    """Mirror the path test_prefill_transformer.py uses for input_source → prompt_text."""
    if input_source in INFINITEBENCH_SUBSETS:
        cached_path = download_infinitebench_subset(input_source)
        with open(cached_path) as f:
            return json.load(f)["prompt"]
    # Add other sources (json_prompts, abc_1k, ...) here if you need them; the
    # test enumerates them at test_prefill_transformer.py:324-348. The minimal
    # path for prefill_runner.py validation is longbook_qa_eng.
    raise ValueError(f"unsupported input_source={input_source!r}. Supported: " f"{sorted(INFINITEBENCH_SUBSETS)}")


def main() -> None:
    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not model_path:
        raise RuntimeError("DEEPSEEK_V3_HF_MODEL must be set")
    cache_root = os.environ.get("TT_DS_PREFILL_HOST_REF_CACHE")
    if not cache_root:
        raise RuntimeError("TT_DS_PREFILL_HOST_REF_CACHE must be set (where to write the .pt)")
    Path(cache_root).mkdir(parents=True, exist_ok=True)

    num_layers = int(os.environ.get("PREFILL_NUM_LAYERS", "1"))
    isl_total = int(os.environ.get("PREFILL_MAX_SEQ_LEN", "1024"))
    input_source = os.environ.get("PREFILL_KV_GOLDEN_INPUT_SOURCE", "longbook_qa_eng")
    padding_side = os.environ.get("PREFILL_KV_GOLDEN_PAD_SIDE", "right")
    weight_type = os.environ.get("PREFILL_KV_GOLDEN_WEIGHT_TYPE", "pretrained")

    logger.info(
        f"[gen-golden] model={model_path} input_source={input_source} "
        f"isl_total={isl_total} num_layers={num_layers} padding_side={padding_side} "
        f"weight_type={weight_type}"
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    n_routed_experts = config.n_routed_experts
    cache_key = ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=n_routed_experts,
        padding_side=padding_side,
    )

    if check_reference_cache_exists(cache_key):
        logger.warning(
            f"[gen-golden] golden already exists for {cache_key}; skipping. "
            f"Delete the .pt file to force regeneration."
        )
        return

    # Tokenize. tokenize_prompt_to_isl always right-pads; if padding_side="left"
    # is requested the test does it differently — see the test's tokenizer
    # fixture if you need left-pad goldens.
    if padding_side != "right":
        raise NotImplementedError(f"gen_kv_golden only supports padding_side='right' currently; got {padding_side!r}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_text = _resolve_prompt_text(input_source)
    token_ids, attention_mask, _ = tokenize_prompt_to_isl(tokenizer, max_isl=isl_total, prompt_text=prompt_text)
    logger.info(
        f"[gen-golden] tokenized: shape={tuple(token_ids.shape)} "
        f"first10={token_ids[0, :10].tolist()} last10={token_ids[0, -10:].tolist()}"
    )

    # Run reference forward layer-by-layer. compute_reference=True populates
    # ref_kvpe_list; build_ttnn_cache=False means no MeshDevice is needed.
    result = load_and_compute_layer_by_layer(
        model_path=Path(model_path),
        config=config,
        num_layers=num_layers,
        token_ids=token_ids,
        attention_mask=attention_mask,
        compute_reference=True,
        build_ttnn_cache=False,
        seq_len=isl_total,
    )

    if result.ref_snapshots is None or result.ref_kvpe_list is None:
        raise RuntimeError(
            "load_and_compute_layer_by_layer returned no reference outputs — check "
            "that compute_reference=True and the HF model loaded successfully."
        )

    save_reference_cache(cache_key, result.ref_snapshots, result.ref_kvpe_list)
    logger.success(
        f"[gen-golden] wrote {cache_key} "
        f"({len(result.ref_snapshots)} snapshots, {len(result.ref_kvpe_list)} KVPE layers)"
    )

    # Pair the golden with a standalone_input.json containing the *exact* tokens
    # the golden was built from. Without this, the runner would read its default
    # JSON (different prompt) and feed mismatched inputs → PCC drops for an
    # input reason, not a model reason. Point the runner at this file via
    # PREFILL_STANDALONE_INPUT and PCC compares apples to apples.
    json_path = Path(cache_root) / f"standalone_input_{cache_key}.json"
    payload = {
        "task_id": 0,
        # Drop padding from the dumped JSON — the runner pads internally to
        # MAX_SEQ_LEN. Keep only the real tokens (the slice the attention
        # mask says is real).
        "token_ids": token_ids[0][: int(attention_mask[0].sum().item())].tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f)
    logger.success(
        f"[gen-golden] wrote standalone_input json: {json_path} " f"({len(payload['token_ids'])} real tokens)"
    )
    logger.info(
        f"[gen-golden] to validate the runner against this golden, launch with:\n"
        f"    PREFILL_KV_VALIDATE=1 \\\n"
        f"    PREFILL_STANDALONE_INPUT={json_path} \\\n"
        f"    PREFILL_MAX_SEQ_LEN={isl_total} \\\n"
        f"    PREFILL_NUM_LAYERS={num_layers} \\\n"
        f"    PREFILL_KV_GOLDEN_INPUT_SOURCE={input_source} \\\n"
        f"    TT_DS_PREFILL_HOST_REF_CACHE={cache_root} \\\n"
        f"    ... (other model env vars) ..."
    )


if __name__ == "__main__":
    main()

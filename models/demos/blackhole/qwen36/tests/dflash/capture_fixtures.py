# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capture torch-reference fixtures for the DFlash drafter PCC tests. Host only, no device.

The drafter cannot run standalone: its inputs are the *target's* residual stream at
``target_layer_ids`` plus the target's token embeddings. Rather than couple the drafter's
bring-up to on-device target changes (hidden-state taps, which are Milestone 2), this
script runs the HF target and the HF reference drafter once on CPU and persists everything
the device test needs.

Run once, then :mod:`test_drafter_pcc` is a pure device test:

    export HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash
    python models/demos/blackhole/qwen36/tests/dflash/capture_fixtures.py

Fixtures land in ``$QWEN36_DFLASH_FIXTURE_DIR`` (default ``generated/dflash_fixtures``,
which is gitignored -- ``target_hidden`` alone is 210 MB at ctx 4096, far past the repo's
500 KB large-file hook, and these are reproducible artifacts, not source).

Two traps this script is written to avoid, both from the Muse-Glimmer DFlash work log:

* **F3** -- ``to_empty()`` / ``load_state_dict(assign=True)`` silently leaves ``inv_freq``
  uninitialised, because it is a *non-persistent* buffer that no checkpoint provides. The
  result is garbage RoPE and a PCC of 0.73-0.92 that *degrades with context*, which reads
  like a genuine long-context bug. So: construct normally and load with ``assign=False``
  (the default), then assert the load was strict.
* **F3b** -- a golden captured through a harness that accidentally dropped the sliding
  window will happily certify an *unwindowed* port at 0.99997 while failing the correct one
  at 0.9294. So this script asserts the window is live on the config it actually builds,
  and lets the reference build its own masks internally rather than passing one in.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.blackhole.qwen36.reference.dflash.dflash import (
    DFlashDraftModel,
    _draft_value,
    _raw_input_embeddings,
    extract_context_feature,
)
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig

DEFAULT_FIXTURE_DIR = "generated/dflash_fixtures"

# Real long-form prose, the same corpus the teacher-forcing e2e test uses. Two reasons it
# is not a short instruction prompt repeated to length:
#
#  * Random or synthetic ids through a speculative drafter measure disagreement between two
#    models on noise, not correctness.
#  * A *repeated* prompt is worse than useless here: repetition is trivial to draft, so
#    candidate agreement comes out inflated. Muse-Glimmer work_log F25 records a published
#    DFlash headline of 83% acceptance that turned out to be measured on a repeating-pattern
#    prompt, against ~20 true tok/s on real content -- "a published headline is a workload
#    before it is a number".
#
# 220 KB of bz2 prose is far more than the 4096 tokens the largest fixture needs, so no
# context length here has to repeat anything.
_TALE_OF_TWO_CITIES = Path(__file__).resolve().parents[5] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

# Used only if the corpus is unavailable (e.g. a sparse checkout).
_FALLBACK_TEXT = (
    "Explain, step by step and in detail, how a modern out-of-order CPU executes a single "
    "load instruction: address generation, TLB lookup, cache hierarchy, store-to-load "
    "forwarding, and what happens on a miss. Then contrast that with how a GPU handles the "
    "same access, and explain why the two designs diverge."
)


def _load_text() -> str:
    """``QWEN36_DFLASH_TEXT_FILE`` -> Tale of Two Cities -> fallback.

    Same precedence as ``tests/e2e/test_teacher_forcing_e2e.py::_load_text``.
    """
    path = os.environ.get("QWEN36_DFLASH_TEXT_FILE")
    if path:
        with open(path) as f:
            return f.read()
    if _TALE_OF_TWO_CITIES.is_file():
        import bz2

        with bz2.open(_TALE_OF_TWO_CITIES, "rt", encoding="utf-8") as f:
            text = f.read()
        # Skip the Gutenberg header and table of contents. Those first few hundred tokens
        # are chapter headings and whitespace runs -- structurally repetitive, which is the
        # very thing this corpus was chosen to avoid. Start at the novel's opening line.
        opening = "It was the best of times"
        if (cut := text.find(opening)) > 0:
            return text[cut:]
        logger.warning("corpus opening line not found; using it from the top (may include front matter)")
        return text
    logger.warning(f"corpus not found at {_TALE_OF_TWO_CITIES}; falling back to a short prompt")
    return _FALLBACK_TEXT


def fixture_dir() -> Path:
    return Path(os.environ.get("QWEN36_DFLASH_FIXTURE_DIR", DEFAULT_FIXTURE_DIR))


def fixture_path(ctx_len: int) -> Path:
    return fixture_dir() / f"drafter_ctx{ctx_len}.pt"


def _resolve(model_id: str) -> str:
    """Local dir as-is, else the snapshot dir for a Hub id."""
    if os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def _drafter_hf_config(drafter_dir: str):
    """The reference's ``Qwen3Config``, built from the drafter's own ``config.json``.

    ``architectures``/``auto_map``/``dtype`` are dropped: the first two would make HF try to
    resolve ``dflash.DFlashDraftModel`` from a repo that ships no ``dflash.py``. Everything
    else -- including ``dflash_config``, ``block_size`` and ``num_target_layers`` -- is
    carried through as config attributes, which is how the reference reads them.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    raw = json.load(open(os.path.join(drafter_dir, "config.json")))
    cfg = Qwen3Config(**{k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "dtype")})

    # F3b guard: assert the window survived config construction. Qwen3Config.__post_init__
    # nulls sliding_window unless use_sliding_window is set, and its default is False -- so
    # a checkpoint (or a hand-built config) that omits the flag silently produces an
    # unwindowed model, and every golden captured from it would be wrong in the one way
    # that is hardest to detect downstream.
    if "sliding_attention" in (cfg.layer_types or []):
        assert cfg.sliding_window, "sliding layers present but sliding_window is None -- golden would be unwindowed"
    return cfg


def _weight_fingerprint(path: str) -> str:
    """Cheap identity for the drafter checkpoint, so a stale fixture cannot pass silently."""
    h = hashlib.sha256()
    with open(os.path.join(path, "config.json"), "rb") as f:
        h.update(f.read())
    st = os.stat(os.path.join(path, "model.safetensors"))
    h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def load_reference_drafter(drafter_dir: str, dtype=torch.float32) -> DFlashDraftModel:
    """The real ``DFlashDraftModel`` with real weights, loaded the safe way (see F3)."""
    from safetensors.torch import load_file

    cfg = _drafter_hf_config(drafter_dir)
    model = DFlashDraftModel(cfg)  # normal init: rotary_emb.inv_freq gets computed here
    state = load_file(os.path.join(drafter_dir, "model.safetensors"))

    # assign=False (the default) copies into the already-initialised parameters, leaving
    # non-persistent buffers like rotary_emb.inv_freq intact. strict=True proves every
    # parameter came from the checkpoint -- a missing key would leave a layer at random
    # init and quietly turn every PCC below into a measurement of noise.
    missing, unexpected = model.load_state_dict(state, strict=False, assign=False)
    assert not missing, f"drafter weights missing from checkpoint: {missing}"
    assert not unexpected, f"checkpoint has weights the drafter does not use: {unexpected}"
    assert torch.isfinite(model.rotary_emb.inv_freq).all(), "inv_freq uninitialised (F3)"

    return model.to(dtype).eval()


@torch.no_grad()
def capture(
    ctx_lens: list[int],
    target_id: str,
    drafter_id: str,
    out_dir: Path,
    gen_tokens: int = 128,
    min_acceptance: int = 2,
) -> None:
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    target_dir, drafter_dir = _resolve(target_id), _resolve(drafter_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    drafter_cfg = DFlashDrafterConfig.from_pretrained(drafter_dir)
    block = drafter_cfg.block_size
    max_ctx = max(ctx_lens)

    tokenizer = AutoTokenizer.from_pretrained(target_dir)
    encoded = tokenizer.encode(_load_text(), add_special_tokens=False)
    assert len(encoded) >= max_ctx, (
        f"corpus yields only {len(encoded)} tokens but ctx_len {max_ctx} was requested; "
        "repeating it would inflate candidate agreement (see F25)"
    )
    logger.info(f"corpus tokenized to {len(encoded)} tokens")

    logger.info(f"loading HF target from {target_dir} (bf16, CPU) ...")
    text_config = Qwen3_5TextConfig.from_pretrained(target_dir)
    target, info = Qwen3_5ForCausalLM.from_pretrained(
        target_dir, config=text_config, dtype=torch.bfloat16, output_loading_info=True
    )
    # Unexpected keys are fine (the composite VLM checkpoint carries visual.*/mtp.*), but a
    # MISSING key means a target weight stayed at random init.
    assert not info["missing_keys"], f"HF target has uninitialized weights: {info['missing_keys']}"
    target.eval()
    drafter_cfg.assert_matches_target(text_config.num_hidden_layers)

    logger.info(f"loading reference drafter from {drafter_dir} ...")
    drafter = load_reference_drafter(drafter_dir)
    embed_scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))

    # A float32 copy of the target's LM head, made ONCE. Do not reach for
    # `target.lm_head.float()`: nn.Module.float() casts in place, which would leave the
    # target's head float32 while the rest of it stays bf16 and blow up the next context
    # length's forward with "expected m1 and m2 to have the same dtype".
    lm_head_weight = target.lm_head.weight.detach().float()

    def output_head(hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden.float(), lm_head_weight)

    for ctx_len in sorted(ctx_lens):
        logger.info(f"--- capturing ctx_len={ctx_len} ---")
        # Positions run over context AND block: the drafter's q takes the last `block` of
        # these while its k/v take all of them (see apply_rotary_pos_emb in the reference).
        position_ids = torch.arange(ctx_len + block).unsqueeze(0)

        # Build the context as a document prompt through the chat template, followed by
        # `gen_tokens` of the target's OWN greedy generation.
        #
        # This shape is load-bearing, and getting it wrong is measurably expensive. The
        # drafter is trained to predict the *target's output distribution*, so what matters
        # is that the block being drafted is text the target itself would produce. Measured
        # on this very checkpoint: drafting a continuation of 19th-century prose accepted
        # 1/15 candidates, while drafting the target's own answer to a prompt accepted a
        # mean of ~5.8/15 (~7.8 tokens committed per block, in line with the gemma-4-31B
        # DFlash reference's published 7.50/iteration). A raw-corpus context is not a
        # broken golden -- it is a pessimistic *workload*, and it would have sent the device
        # port hunting a numerics bug that does not exist.
        #
        # A long document keeps the context real and non-repetitive at any length while
        # costing only `gen_tokens` decode steps, so ctx 4096 is as cheap to build as 512.
        doc_budget = max(0, ctx_len - gen_tokens - 64)  # 64 covers chat-template overhead
        if doc_budget:
            doc = tokenizer.decode(encoded[:doc_budget])
            user_msg = f"{doc}\n\nBased on the text above, summarize what has happened so far and why it matters."
        else:
            user_msg = "How many positive whole-number divisors does 196 have? Think step by step."
        prompt_ids = tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
            ),
            return_tensors="pt",
        ).input_ids.to(torch.long)

        # Trim the document (never pad) so prompt + generation lands exactly on ctx_len.
        overshoot = prompt_ids.shape[1] + gen_tokens - ctx_len
        if overshoot > 0:
            assert doc_budget, f"short prompt overshoots ctx_len={ctx_len}; raise it or lower --gen-tokens"
            doc = tokenizer.decode(encoded[: doc_budget - overshoot])
            user_msg = f"{doc}\n\nBased on the text above, summarize what has happened so far and why it matters."
            prompt_ids = tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
                ),
                return_tensors="pt",
            ).input_ids.to(torch.long)
        gen_steps = ctx_len - prompt_ids.shape[1]
        assert gen_steps >= 0, f"prompt is {prompt_ids.shape[1]} tokens, longer than ctx_len={ctx_len}"
        logger.info(f"  prompt {prompt_ids.shape[1]} tokens + {gen_steps} self-generated = {ctx_len}")

        out = target(
            prompt_ids,
            position_ids=position_ids[:, : prompt_ids.shape[1]],
            use_cache=True,
            output_hidden_states=True,
        )
        # `hidden_states[i + 1]` is the OUTPUT of decoder layer i; index 0 is the embedding.
        # extract_context_feature applies that +1 offset itself and concatenates the taps in
        # target_layer_ids order, which `fc` depends on.
        taps = [extract_context_feature(out.hidden_states, drafter.target_layer_ids).float()]
        cache = out.past_key_values
        tok_id = int(out.logits[0, -1].float().argmax())
        seq = prompt_ids[0].tolist()

        # Greedy self-generation, accumulating taps per step. Using the KV cache here is
        # safe: generation only moves forward, so the hybrid model's GDN recurrent state
        # never needs the rollback that the speculative loop itself will.
        for step in range(gen_steps):
            seq.append(tok_id)
            out = target(
                torch.tensor([[tok_id]], dtype=torch.long),
                position_ids=position_ids[:, len(seq) - 1 : len(seq)],
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            taps.append(extract_context_feature(out.hidden_states, drafter.target_layer_ids).float())
            cache = out.past_key_values
            tok_id = int(out.logits[0, -1].float().argmax())

        target_hidden = torch.cat(taps, dim=1)
        input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        assert input_ids.shape[1] == ctx_len, (input_ids.shape, ctx_len)
        assert target_hidden.shape == (1, ctx_len, drafter_cfg.target_feature_size), target_hidden.shape
        logger.info(f"  context tail: {tokenizer.decode(seq[-80:])!r}")

        # The anchor is the token the target itself commits next; block slot 0 holds it and
        # the remaining `block - 1` slots start at the absorbing mask token.
        prefill_cache = cache
        anchor = tok_id
        del taps, out
        block_ids = torch.full((1, block), drafter_cfg.mask_token_id, dtype=torch.long)
        block_ids[0, 0] = anchor
        noise_embedding = _raw_input_embeddings(target, block_ids, embed_scale).float()
        gc.collect()

        # attention_mask stays None so the reference builds its own per-layer masks (F3b);
        # past_key_values stays None because a fresh DynamicCache's update() is the identity
        # here, and _make_cache would drag in transformers 5.15's activate_past_recording().
        hidden = drafter(
            position_ids=position_ids,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
        )
        assert hidden.shape == (1, block, drafter_cfg.hidden_size), hidden.shape

        # The reference keeps the last block-1 rows: slot 0 is the anchor, and row i
        # predicts candidate i.
        draft_hidden = hidden[:, 1 - block :, :]
        logits = drafter.compute_logits(draft_hidden, output_head)
        candidates = logits.argmax(dim=-1)[0]
        assert candidates.shape == (drafter_cfg.num_draft_tokens,), candidates.shape

        # Validate the golden itself, not just its shapes (F3b). Run the target over the
        # verify block exactly as dflash_generate would and count how many leading
        # candidates it agrees with. This is the discriminator the Muse-Glimmer port's
        # dflash_device_e2e.py is built around: shapes and PCC can be perfect while the
        # conditioning (taps, tap order, positions, anchor) is wrong, and the only symptom
        # is that acceptance collapses to ~0. A fixture that fails this would send the
        # device port chasing a nonexistent numerics bug.
        verify_ids = torch.cat([torch.tensor([[anchor]], dtype=torch.long), candidates[None].long()], dim=1)
        verify_out = target(
            verify_ids,
            position_ids=position_ids[:, ctx_len : ctx_len + block],
            past_key_values=prefill_cache,
            use_cache=False,
        )
        target_argmax = verify_out.logits[0].float().argmax(dim=-1)
        matches = int((candidates == target_argmax[:-1]).to(torch.int32).cumprod(0).sum())
        logger.info(
            f"  golden acceptance: {matches}/{drafter_cfg.num_draft_tokens} leading candidates "
            f"accepted by the target ({matches + 1} tokens committed per block)"
        )
        assert matches >= min_acceptance, (
            f"golden accepts only {matches} candidates (floor {min_acceptance}). Shapes and PCC can "
            f"be perfect while the conditioning is wrong -- check tap order, tap offset, the anchor, "
            f"and that the drafted block is in the target's own output distribution."
        )
        del verify_out, prefill_cache
        gc.collect()

        payload = {
            "ctx_len": ctx_len,
            "block_size": block,
            "target_hidden": target_hidden.to(torch.bfloat16),
            "noise_embedding": noise_embedding.to(torch.bfloat16),
            "position_ids": position_ids,
            "input_ids": input_ids,
            "anchor_token_id": anchor,
            "reference_hidden": hidden.float(),
            "reference_draft_hidden": draft_hidden.float(),
            "reference_candidates": candidates,
            "golden_acceptance": matches,
            "target_argmax": target_argmax,
            "drafter_fingerprint": _weight_fingerprint(drafter_dir),
            "target_model": target_id,
            "drafter_model": drafter_id,
        }
        path = fixture_path(ctx_len)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info(
            f"wrote {path} ({path.stat().st_size / 2**20:.1f} MiB); "
            f"anchor={anchor} candidates={candidates.tolist()[:8]}..."
        )
        logger.info(f"  decoded: {tokenizer.decode([anchor] + candidates.tolist())!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ctx-len",
        type=int,
        nargs="+",
        default=[512, 4096],
        help="context lengths to capture. Include one > sliding_window (2048): it is the "
        "only regime where the window's lower bound is exercised at all.",
    )
    ap.add_argument("--target", default=os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--drafter", default=os.environ.get("DFLASH_HF_MODEL", "z-lab/Qwen3.6-27B-DFlash"))
    ap.add_argument(
        "--gen-tokens",
        type=int,
        default=128,
        help="tokens of the target's own greedy generation to append to the document prompt. "
        "The block being drafted must sit in the target's output distribution or acceptance "
        "collapses (~1/15 on raw prose vs ~5.8/15 in-distribution), and this costs the same "
        "number of decode steps at any ctx_len.",
    )
    ap.add_argument(
        "--min-acceptance",
        type=int,
        default=2,
        help="fail if the golden accepts fewer leading candidates than this. Guards against "
        "a silently mis-conditioned fixture; set 0 to capture anyway.",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    capture(
        args.ctx_len,
        args.target,
        args.drafter,
        args.out_dir or fixture_dir(),
        args.gen_tokens,
        args.min_acceptance,
    )


if __name__ == "__main__":
    main()

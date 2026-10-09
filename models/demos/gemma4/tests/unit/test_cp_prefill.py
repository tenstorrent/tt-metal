# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CP prefill (galaxy one-instance): engagement, needle retrieval through multi-chunk prefill, and the
short-prompt fallback; a per-column offset or gather bug across the lanes loses the needle."""

import os

import torch

from ...tests.test_factory import parametrize_mesh_with_fabric

GEN_TOKENS = 14
FILLER = "The sky is blue and the grass is green. "


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_cp_prefill(mesh_device, reset_seeds, request):
    os.environ["GEMMA4_GALAXY_FRACTURE"] = "1"
    os.environ["GEMMA4_CP_PREFILL"] = "1"
    os.environ.pop("GEMMA4_GALAXY_LANES", None)  # CP and lanes are exclusive
    # No chunk override: the CP-paired default (24576, via
    # gemma4_cp_prefill_engaged) must kick in — the production pairing.

    from models.demos.gemma4.tt.attention import operations as attn_ops
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")

    blocks_per_user = 514  # 32896 tokens: ~26K prompt + generation
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=1 + blocks_per_user)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=32,
        max_seq_len=65536,
        paged_attention_config=paged_cfg,
    )
    assert getattr(generator.model[0].mesh_config, "cp_prefill", False), "CP gate did not engage"
    block_ids = torch.arange(1, 1 + blocks_per_user, dtype=torch.int32)

    def ask(content, expect, min_len=0):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and isinstance(ids[0], str):
            ids = tokenizer.convert_tokens_to_ids(ids)
        assert len(ids) >= min_len, f"prompt too short to exercise chunking: {len(ids)} < {min_len}"
        t = torch.tensor(ids, dtype=torch.long)
        plen = int(t.shape[-1])
        padded = 1 << max(int(plen - 1).bit_length(), 7)
        tok1 = torch.zeros(1, padded, dtype=torch.long)
        tok1[0, :plen] = t
        out = generator.prefill_forward_text(
            tok1,
            page_table=block_ids.unsqueeze(0),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([plen]),
            empty_slots=[0],
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        cur = int(torch.argmax(lg.reshape(-1)).item())
        toks, pos = [], plen
        for _ in range(GEN_TOKENS):
            logits = generator.decode_forward(
                torch.tensor([[cur]], dtype=torch.long),
                torch.tensor([pos], dtype=torch.long),
                page_table=block_ids.unsqueeze(0),
                kv_cache=tt_kv_cache,
                enable_trace=False,
                read_from_device=True,
                sampling_params=None,
            )
            l0 = logits[0] if isinstance(logits, (list, tuple)) else logits
            cur = int(torch.argmax(l0.reshape(-1)).item())
            toks.append(cur)
            pos += 1
        answer = tokenizer.decode(toks, skip_special_tokens=True)
        print(f"[cp] plen={plen} expect={expect!r} -> {answer!r}")
        assert expect in answer.lower(), f"expected {expect!r} in {answer!r}"
        return plen

    assert int(generator.model_args[0].max_prefill_chunk_size) == 24576, (
        "CP-paired chunk default did not resolve: " f"{generator.model_args[0].max_prefill_chunk_size}"
    )

    # ~26K tokens -> two CP chunks; the needle sits in the first and the question in the second,
    # so a per-column offset or gather bug (wrong prefix visibility) loses the needle.
    filler = " ".join(f"Entry {i}: shipment {i} arrived at dock {i % 40} on day {i % 28}." for i in range(1150))
    needle_pos = len(filler) // 4
    long_content = (
        filler[:needle_pos]
        + " The secret access code is 7291. "
        + filler[needle_pos:]
        + " What is the secret access code? Answer with just the number."
    )
    ask(long_content, "7291", min_len=24700)
    assert getattr(
        attn_ops.chunked_prefill_sdpa, "_cp_logged", False
    ), "CP branch never engaged on the aligned long prompt"

    # A short prompt stays single-chunk and must still answer correctly.
    ask("What is the opposite of hot? Answer in one word.", "cold")

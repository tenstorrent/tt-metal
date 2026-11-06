# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: DFlash2 speculative decoding end to end (52-layer target) vs teacher-forced plain greedy decode."""

import os
from collections import Counter

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.muse_glimmer_30b.demo.text_demo import _DFLASH_BLOCK, _dflash_generator, _isl_prompt
from models.demos.muse_glimmer_30b.tt.mesh_utils import mesh_test_params, skip_and_submesh
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, checkpoint_dir, dflash_checkpoint_dir
from models.demos.muse_glimmer_30b.tt.tokenizer_utils import decode, load_tokenizer

pytestmark = pytest.mark.skipif(
    not (
        os.path.exists(os.path.join(checkpoint_dir(), "config.json"))
        and os.path.exists(os.path.join(dflash_checkpoint_dir(), "model.safetensors"))
    ),
    reason="target or DFlash2 checkpoint not available",
)

_mesh_params = mesh_test_params()
MAX_NEW = 128


@pytest.mark.timeout(3600)
@_mesh_params
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("isl", [128, 8192], ids=["isl128", "isl8k"])
def test_dflash2_e2e(mesh_device, tp, enable_trace, isl):
    if isl == 8192 and not enable_trace:
        pytest.skip("isl8k runs the trace variant only")
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.verify import set_greedy_sampling
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    tokenizer = load_tokenizer(checkpoint_dir())
    prompt = _isl_prompt(tokenizer, isl, reserve=MAX_NEW + _DFLASH_BLOCK)
    S = len(prompt)
    max_seq_len = -(-(S + MAX_NEW + _DFLASH_BLOCK) // 256) * 256
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sub)
    gen = _dflash_generator(sub, args, model, enable_trace=enable_trace)
    nb, pt, pt_tt = args.max_num_blocks_per_user, gen.page_table, gen.page_table_tt
    tokens, stats = gen.generate(prompt, MAX_NEW)

    gen.release_traces()

    # ---- teacher-forced plain greedy decode over the speculative output
    set_greedy_sampling(model)
    model.prefill_forward(torch.tensor(prompt, dtype=torch.int64), 0, pt_tt)
    dev_in = model.allocate_decode_inputs(1, nb)
    plain = []
    for i in range(len(tokens) - 1):
        host = model.prepare_decode_inputs_host(torch.tensor([tokens[i]]), torch.tensor([S + i]), pt)
        model.copy_decode_inputs(host, dev_in)
        logits = model.decode_forward(*dev_in)
        tok = model.sample_on_device(logits)
        plain.append(int(model.tokens_to_torch(tok, 1)[0]))
        ttnn.deallocate(tok)
        if plain[-1] != tokens[i + 1]:  # diagnose: is the mismatch a near-tie of the plain-path logits?
            lg = model.logits_to_torch(logits, 1)[0]
            top = torch.topk(lg, 2)
            logger.info(
                f"mismatch at {i}: plain top-2 ids={top.indices.tolist()} logits={[round(v, 4) for v in top.values.tolist()]} "
                f"gap={float(top.values[0] - top.values[1]):.4f} | speculative token {tokens[i + 1]} "
                f"plain logit={float(lg[tokens[i + 1]]):.4f}"
            )
        ttnn.deallocate(logits)
    hits = [plain[i] == tokens[i + 1] for i in range(len(plain))]
    agreement = sum(hits) / max(1, len(hits))
    first_div = next((i for i, h in enumerate(hits) if not h), None)

    hist = dict(sorted(Counter(stats["accept_lengths"]).items()))
    logger.info(f"prompt tokens={S} generated={len(tokens)}")
    logger.info(f"agreement={agreement:.4f} first_divergence={first_div}")
    logger.info(f"mean_accept={stats['mean_accept']:.2f} rounds={stats['rounds']} accept_hist={hist}")
    logger.info(f"ttft={stats['ttft_s']:.2f}s decode={stats['decode_tok_s']:.2f} tok/s (eager)")
    logger.info(
        f"stats: {stats['mean_ms']} decode_tok_s(from round 2)={stats['decode_tok_s']:.2f} all={stats['decode_tok_s_all']:.2f}"
    )
    logger.info(f"text={decode(tokenizer, tokens)[:300]!r}")
    logger.info(f"E2E agreement={agreement:.4f} first_divergence={first_div} mean_accept={stats['mean_accept']:.2f}")
    gen.close()
    assert agreement >= 0.95, f"agreement {agreement:.3f}, first divergence at {first_div}"
    assert stats["mean_accept"] >= 3.0, f"mean accept {stats['mean_accept']:.2f}"


@pytest.mark.timeout(3600)
@_mesh_params
def test_dflash2_e2e_sampling(mesh_device, tp):
    """Sampling mode (host rejection sampling): seed determinism, seed sensitivity, near-greedy plumbing check."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    tokenizer = load_tokenizer(checkpoint_dir())
    prompt = _isl_prompt(tokenizer, 128, reserve=MAX_NEW + _DFLASH_BLOCK)
    S = len(prompt)
    max_seq_len = -(-(S + MAX_NEW + _DFLASH_BLOCK) // 256) * 256
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sub)
    gen = _dflash_generator(sub, args, model, enable_trace=True)

    greedy_tokens, greedy_stats = gen.generate(prompt, MAX_NEW)

    default = args.default_sampling
    assert default is not None, "generation_config has no default sampling"
    tok0, st0 = gen.generate(prompt, MAX_NEW, sampling=default, seed=0)
    tok0b, _ = gen.generate(prompt, MAX_NEW, sampling=default, seed=0)
    tok1, _ = gen.generate(prompt, MAX_NEW, sampling=default, seed=1)
    logger.info(
        f"sampling={st0['sampling']} mean_accept={st0['mean_accept']:.2f} decode={st0['decode_tok_s']:.2f} tok/s"
    )
    logger.info(f"text(seed 0)={decode(tokenizer, tok0)[:300]!r}")

    near = {"temperature": 1e-3, "top_p": 1.0, "top_k": 32}
    tok_n, st_n = gen.generate(prompt, MAX_NEW, sampling=near, seed=0)
    n = min(len(tok_n), len(greedy_tokens))
    agreement = sum(a == b for a, b in zip(tok_n[:n], greedy_tokens[:n])) / max(1, n)
    ratio = st_n["mean_accept"] / max(1e-9, greedy_stats["mean_accept"])
    logger.info(
        f"near-greedy agreement={agreement:.4f} mean_accept={st_n['mean_accept']:.2f} "
        f"greedy mean_accept={greedy_stats['mean_accept']:.2f} ratio={ratio:.3f}"
    )
    gen.release_traces()
    gen.close()
    assert tok0 == tok0b, "same seed gave different tokens"
    assert tok0 != tok1, "different seeds gave identical tokens"
    assert st0["mean_accept"] >= 1.5, f"mean accept {st0['mean_accept']:.2f}"
    assert agreement >= 0.95, f"near-greedy agreement {agreement:.3f}"
    assert (
        abs(ratio - 1.0) <= 0.15
    ), f"near-greedy mean_accept {st_n['mean_accept']:.2f} vs greedy {greedy_stats['mean_accept']:.2f}"


@pytest.mark.timeout(3600)
@_mesh_params
def test_dflash2_step_api(mesh_device, tp):
    """vLLM-style step loop (start / verify / commit / propose) with host greedy acceptance vs generate()."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    tokenizer = load_tokenizer(checkpoint_dir())
    prompt = _isl_prompt(tokenizer, 128, reserve=MAX_NEW + _DFLASH_BLOCK)
    S = len(prompt)
    max_seq_len = -(-(S + MAX_NEW + _DFLASH_BLOCK) // 256) * 256
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sub)
    gen = _dflash_generator(sub, args, model, enable_trace=True)

    ref_tokens, _ = gen.generate(prompt, MAX_NEW)

    lg = gen.start(prompt, return_logits=True)
    t = int(lg.argmax())  # vLLM samples the first token
    out = [t]
    lg = gen.verify([t], "logits")  # ordinary decode
    t1 = int(lg[0].argmax())
    gen.commit(1, t1)
    out.append(t1)
    accepted = []
    n_verify = 0
    while len(out) < MAX_NEW and out[-1] not in gen.stop_ids:
        if gen.p + gen.rows > gen.model.args.max_seq_len:
            break
        drafts, _, _ = gen.propose()
        if n_verify % 3 == 2:
            am = gen.verify([gen.anchor] + drafts, "logits").argmax(-1).tolist()
        else:
            am = gen.verify([gen.anchor] + drafts, "argmax")
        n_verify += 1
        acc = 0
        while acc < len(drafts) and drafts[acc] == am[acc]:
            acc += 1
        bonus = am[acc]
        new = drafts[:acc] + [bonus]
        gen.commit(acc + 1, bonus)
        accepted.append(acc + 1)
        for i, x in enumerate(new):
            if x in gen.stop_ids:
                new = new[: i + 1]
                break
        out += new
    out = out[:MAX_NEW]
    n = min(len(out), len(ref_tokens))
    hits = [a == b for a, b in zip(out[:n], ref_tokens[:n])]
    agreement = sum(hits) / max(1, n)
    first_div = next((i for i, h in enumerate(hits) if not h), None)
    mean_acc = sum(accepted) / max(1, len(accepted))
    logger.info(
        f"step API agreement={agreement:.4f} first_divergence={first_div} mean_accepted_per_verify={mean_acc:.2f}"
    )
    if first_div is not None:
        logger.info(
            f"divergence: step token {out[first_div]} vs generate token {ref_tokens[first_div]} (top-2 gap not available in step loop)"
        )
    gen.release_traces()
    gen.close()
    assert agreement >= 0.99, f"agreement {agreement:.3f}, first divergence at {first_div}"


@pytest.mark.timeout(3600)
@_mesh_params
def test_dflash2_decline_then_steps(mesh_device, tp):
    """A vLLM-style step loop (no warmup; lazy compile then capture) with one declined propose matches generate()."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    tokenizer = load_tokenizer(checkpoint_dir())
    prompt = _isl_prompt(tokenizer, 128, reserve=MAX_NEW + _DFLASH_BLOCK)
    S = len(prompt)
    max_seq_len = -(-(S + MAX_NEW + _DFLASH_BLOCK) // 256) * 256
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=1, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sub)
    gen = _dflash_generator(sub, args, model, enable_trace=True)

    def run(decline_at):
        lg = gen.start(prompt, return_logits=True)
        t = int(lg.argmax())
        out = [t]
        lg = gen.verify([t], "logits")  # ordinary decode
        t1 = int(lg[0].argmax())
        gen.commit(1, t1)
        out.append(t1)
        accepted = []
        n_round = 0
        while len(out) < MAX_NEW and out[-1] not in gen.stop_ids:
            if gen.p + gen.rows > gen.model.args.max_seq_len:
                break
            if n_round == decline_at:  # the adapter declined propose: an ordinary decode verifies the anchor alone
                lg = gen.verify([gen.anchor], "logits")
                tn = int(lg[0].argmax())
                gen.commit(1, tn)
                out.append(tn)
                accepted.append(1)
                n_round += 1
                continue
            drafts, _, _ = gen.propose()
            if n_round % 3 == 2:
                am = gen.verify([gen.anchor] + drafts, "logits").argmax(-1).tolist()
            else:
                am = gen.verify([gen.anchor] + drafts, "argmax")
            n_round += 1
            acc = 0
            while acc < len(drafts) and drafts[acc] == am[acc]:
                acc += 1
            bonus = am[acc]
            new = drafts[:acc] + [bonus]
            gen.commit(acc + 1, bonus)
            accepted.append(acc + 1)
            for i, x in enumerate(new):
                if x in gen.stop_ids:
                    new = new[: i + 1]
                    break
            out += new
        return out[:MAX_NEW], accepted

    ref_tokens, _ = gen.generate(prompt, MAX_NEW)
    out, accepted = run(decline_at=None)
    n = min(len(out), len(ref_tokens))
    agreement = sum(a == b for a, b in zip(out[:n], ref_tokens[:n])) / max(1, n)
    logger.info(f"plain steps agreement={agreement:.4f} mean_accepted={sum(accepted) / max(1, len(accepted)):.2f}")

    out2, acc2 = run(decline_at=4)
    k = 4  # rounds before the declined propose
    before = sum(acc2[:k]) / max(1, k)
    after = sum(acc2[k + 1 :]) / max(1, len(acc2) - k - 1)
    n2 = min(len(out2), len(ref_tokens))
    agree2 = sum(a == b for a, b in zip(out2[:n2], ref_tokens[:n2])) / max(1, n2)
    logger.info(
        f"decline scenario: completed {len(out2)} tokens, mean accepted before={before:.2f} after={after:.2f} "
        f"agreement={agree2:.4f}"
    )
    gen.release_traces()
    gen.close()
    assert agreement >= 0.99, f"agreement {agreement:.3f}"
    assert agree2 >= 0.99, f"decline agreement {agree2:.3f}"
    assert len(out2) >= MAX_NEW or out2[-1] in gen.stop_ids


def _first_diff(got, want):
    d = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    return min(len(got), len(want)) if d is None and len(got) != len(want) else d


BATCH_NEW = 512


@pytest.mark.timeout(5400)
@_mesh_params
def test_dflash2_batch(mesh_device, tp):
    """Several users per step (K by user count): each user's greedy tokens equal the one-user batch tokens of its
    prompt (2, 4 and 8 users; logs the total decode rate). A sampled 4-user run repeats with the same seeds."""
    sub = skip_and_submesh(mesh_device, tp)
    from models.demos.muse_glimmer_30b.tt.dflash.batch_generator import build_batch_generator
    from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel

    tokenizer = load_tokenizer(checkpoint_dir())
    prompts = {isl: _isl_prompt(tokenizer, isl, reserve=BATCH_NEW + _DFLASH_BLOCK) for isl in (128, 8192)}
    max_seq_len = -(-(max(len(p) for p in prompts.values()) + BATCH_NEW + _DFLASH_BLOCK) // 256) * 256
    args = MuseGlimmerModelArgs.from_checkpoint(
        mesh_device=sub, max_batch_size=8, max_seq_len=max_seq_len, max_prefill_len=max_seq_len
    )
    model = MuseGlimmerModel(args, sub)
    gen = build_batch_generator(model, 8, enable_trace=True)
    # References: one user on the batch path. The batch path runs SDPA on 32 rows and the single-user path on 16, so
    # their reduction orders differ and a near-tie can flip a late token: the single path is only compared in the log.
    ref = {}
    for isl, prompt in prompts.items():
        single, _ = gen.generate(prompt, BATCH_NEW)
        (ref[isl],), st = gen.generate_batch([prompt], BATCH_NEW)
        d, gap = _first_diff(ref[isl], single), None
        if d is not None:
            ctx = torch.tensor(prompt + ref[isl][:d], dtype=torch.int64)
            if model.prefill_is_new(ctx.numel()):
                gen.release_traces(keep_seen=True)
            lg_tt = model.prefill_forward_logits_tt(ctx, 0, gen.page_table_tt)
            top = torch.topk(model.logits_to_torch(lg_tt, 1)[0].float(), 2)
            ttnn.deallocate(lg_tt)
            gap = round(float(top.values[0] - top.values[1]), 4)
        logger.info(
            f"BATCHREF isl={isl}: one-user batch vs single path first_diff={d} (prefill top-2 gap there {gap}), "
            f"mean_accept={st['mean_accept'][0]:.2f}"
        )
    order = [128, 8192] * 4
    diffs = {}
    for users in (2, 4, 8):
        isls = order[:users]
        tokens, stats = gen.generate_batch([prompts[i] for i in isls], BATCH_NEW)
        for u, isl in enumerate(isls):
            got, want = tokens[u], ref[isl]
            first = _first_diff(got, want)
            diffs[(users, u)] = first
            logger.info(
                f"BATCHCHECK users={users} slot={u} isl={isl} equal={first is None} first_diff={first} "
                f"len={len(got)}/{len(want)} mean_accept={stats['mean_accept'][u]:.2f}"
            )
        logger.info(
            f"BATCHRESULT users={users} R={stats['R']} full_tok_s={stats['full_tok_s']:.1f} "
            f"full_rounds={stats['full_rounds']}/{stats['rounds']} all_tok_s={stats['decode_tok_s_all']:.1f} "
            f"ms={stats['mean_ms']}"
        )
    sampling = dict(temperature=1.0, top_p=0.95, top_k=32)
    runs = [
        gen.generate_batch([prompts[i] for i in order[:4]], 128, sampling=sampling, seeds=[11, 12, 13, 14])
        for _ in range(2)
    ]
    repeat = runs[0][0] == runs[1][0]
    logger.info(
        f"BATCHSAMPLED users=4 repeat_equal={repeat} mean_accept={[round(a, 2) for a in runs[0][1]['mean_accept']]} "
        f"full_tok_s={runs[0][1]['full_tok_s']:.1f}"
    )
    gen.close()
    bad = {k: v for k, v in diffs.items() if v is not None}
    assert not bad, f"greedy divergence (users, slot) -> first diff: {bad}"
    assert repeat, "the sampled batch run does not repeat with the same seeds"
    assert min(runs[0][1]["mean_accept"]) >= 2.0, runs[0][1]["mean_accept"]

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase B on a MESH (python_env, TT): data-parallel XTTS-v2, one request per chip.

The mesh counterpart of `phase_tt.py`. Target serving shape from `CLAUDE_XTTS_GPT.md`
("Mesh / data-parallel serving"): **pure DP, one request per chip, batch N on a 1xN mesh**.
The models are small enough to fit per chip, so there is no tensor-parallelism and zero
cross-chip communication.

Two DP flavors, selected with `--flavor`:

* `spmd` (default) — **one model instance on the full 1xN mesh**, requests sharded on dim 0
  (`ttnn.shard_tensor_to_mesh_mapper(mesh, 0)`), weights replicated. A single
  `execute_trace` per decode step fans out to all N chips, so host dispatch cost is paid once
  per step instead of N times. This needs no change to any block's tensor shapes because
  **ttnn ops see per-device shard shapes, not the global shape**: an `[N, ...]` host tensor
  sharded on dim 0 arrives at every op as `[1, ...]`, exactly the batch-1 shape the blocks
  were PCC-validated with. (Verified on this machine: a `[32,1,32,32]` torch tensor sharded
  dim-0 across a 1x32 mesh reports `shape == (1,1,32,32)`.)

* `replicas` — **N independent instances on (1,1) submeshes** (`create_submeshes`), the same
  approach `models/tt_transformers/tt/generator.py` uses for data parallelism. Each replica is
  bit-for-bit the single-card path (`mesh_replicate_mapper` is a no-op on a 1x1 submesh) and is
  driven round-robin: every step issues `execute_trace` on all N chips *before* reading any
  result back, so the chips still overlap under one host thread. More host dispatch than
  `spmd`, but the requests are genuinely independent (no lockstep), which is the shape
  continuous batching wants.

Per-request state (cache position, `seen` set for the repetition penalty, RNG, stop flag) is
per slot in both flavors, so requests stop at different steps. In `spmd` a finished slot is
still issued (one lockstep trace) but its position is frozen and its output discarded; in
`replicas` it simply stops being issued.

Usage (tt-metal python_env):

    XTTS_CKPT=/path/to/xtts_ref/model.pth python phase_tt_mesh.py \
        --work ./out/reqs/r0 --replicas 32 --flavor spmd --same-seed

`--work` may be given once (that request is replicated onto every chip — the mesh correctness
check, use with --same-seed) or N times (N distinct requests, the real DP case).

Padding caveats for `spmd` (a lockstep trace needs uniform shapes):
* prompts are right-padded to a common P — safe, see `TTNNGPTTracedDecoder.prefill`;
* Blocks 1/2 consume the *reference clip's* mel, so requests sharing one reference clip need
  no padding at all. Distinct reference clips of different lengths would need per-chip masks
  (the conditioning encoder's `time_mask`/`key_mask` are replicated today) — not implemented,
  the script asserts the shapes match;
* the vocoder pads `z` to a common L and trims each waveform afterwards, so a padded request's
  final samples see zero context beyond its own end (a tail-only difference).
"""

import argparse
import os
import time

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    LATENTS,
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder
from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import TTNNSpeakerEncoder, preprocess_speaker_parameters

GPT_MAX_AUDIO = 605  # model cap: mel_pos is [608,1024]
TEMPERATURE, TOP_K, TOP_P = 0.75, 50, 0.85  # coqui Xtts.inference defaults
AR_COMP, HOP, ISR, OSR = 1024, 256, 22050, 24000  # HifiDecoder.forward constants


# ---------------------------------------------------------------------------------------
# Blocks 1, 2 and 4 — single-shot forwards. `mapper`/`composer` are None for a 1x1 submesh
# (single request) or the dim-0 shard/concat pair for a batched SPMD call.
# ---------------------------------------------------------------------------------------
def block2_speaker_embedding(dev, logmel, mapper=None, composer=None):
    """Block 2 (ResNet speaker encoder): logmel [B,64,T] -> d-vector [B,512,1]."""
    model = TTNNSpeakerEncoder(dev, preprocess_speaker_parameters(dev))
    logmel_tt = ttnn.from_torch(logmel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mapper)
    emb = model(logmel_tt)
    if isinstance(emb, tuple):
        emb = emb[0]
    out = ttnn.to_torch(emb, mesh_composer=composer) if composer else ttnn.to_torch(emb)
    return out.to(torch.float32).reshape(-1, 512, 1)


def block1_gpt_cond_latent(dev, mel, mapper=None, composer=None):
    """Block 1 (conditioning encoder + Perceiver): mel [B,80,T] -> [B,32,1024]."""
    T = mel.shape[2]
    S = ((T + 31) // 32) * 32
    mel_f = torch.nn.functional.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))  # [B,S,80]
    enc = TTNNConditioningEncoder(dev, preprocess_encoder_parameters(dev, dtype=ttnn.float32), t_real=T, s_pad=S)
    perc = TTNNPerceiver(dev, preprocess_perceiver_parameters(dev, dtype=ttnn.float32))
    # The masks are constants of (T, S) — identical for every request here, so replicate them.
    km = torch.zeros(1, 1, 1, LATENTS + S)
    km[:, :, :, LATENTS + T :] = -1e9
    km_tt = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=enc.mesh_mapper)
    mel_tt = ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mapper)
    frames = enc(mel_tt)
    out = perc(frames, km_tt)
    out = ttnn.to_torch(out, mesh_composer=composer) if composer else ttnn.to_torch(out)
    return out.to(torch.float32)  # [B,32,1024]


def block4_vocoder(dev, gpt_latents, spk, mapper=None, composer=None):
    """Block 4 (HiFi-GAN): GPT latents [B,T,1024] + d-vector [B,512,1] -> waveform [B,1,N].

    The two linear time-resizes that HifiDecoder.forward does stay on the host (cheap)."""
    import torch.nn.functional as F

    from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
        TTNNHifiganGenerator,
        preprocess_hifigan_parameters,
    )

    z = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=AR_COMP / HOP, mode="linear")
    z = F.interpolate(z, scale_factor=OSR / ISR, mode="linear")  # [B,1024,L]
    B, _, L = z.shape
    z_nhwc = z.permute(0, 2, 1).reshape(B, 1, L, 1024)
    voc = TTNNHifiganGenerator(dev, preprocess_hifigan_parameters(dev))
    z_tt = ttnn.from_torch(z_nhwc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=mapper)
    g_tt = ttnn.from_torch(
        spk.reshape(B, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mapper
    )
    wav = voc(z_tt, g_tt)
    wav = ttnn.to_torch(wav, mesh_composer=composer) if composer else ttnn.to_torch(wav)
    return wav.to(torch.float32).reshape(B, 1, -1)


# ---------------------------------------------------------------------------------------
# Per-request host state (identical in both flavors)
# ---------------------------------------------------------------------------------------
class RequestState:
    """One request's host-side decode state: prompt, sampler, codes, latents, stop flag."""

    def __init__(self, idx, work, heads, seed):
        self.idx = idx
        self.work = work
        self.heads = heads
        self.codes = []
        self.latents = []
        self.done = False
        self.gen = torch.Generator().manual_seed(seed)

        meta = torch.load(os.path.join(work, "gen_meta.pt"))
        self.start_audio, self.stop_audio = meta["start_audio"], meta["stop_audio"]
        self.penalty = meta["repetition_penalty"]
        self.seen = {self.start_audio}

        self.prefix = torch.load(os.path.join(work, "prefix_emb.pt")).float()  # [1,P,1024]
        self.P = self.prefix.shape[1]
        self.pos = self.P  # first decode step (START_AUDIO) sits at position P
        # START_AUDIO's embedding is the first thing fed to the decoder.
        self.next_emb = (heads["mel_emb"][self.start_audio] + heads["mel_pos"][0]).view(1, 1, -1)

    def sample(self, latent):
        """coqui's decode strategy: repetition penalty -> temperature -> top-k -> top-p ->
        multinomial (do_sample=True), with this request's own RNG."""
        mh_w, mh_b = self.heads["mel_head_w"], self.heads["mel_head_b"]
        logits = (latent @ mh_w.t() + mh_b)[0, 0].clone().float()  # [1026]
        # Repetition penalty (HF applies this first), vectorized. A Python loop over `seen` here
        # is O(len(seen)) per slot per step, and `seen` grows by one token per step — so it makes
        # decode cost grow *linearly with the utterance* and the total quadratic. Measured on a
        # 1x32 mesh at 586 steps: the loop version ran 90.6 ms/step mean, rising 30.7 -> 128.1
        # ms/step, versus 8.9 ms/step for the device alone. Indexing once is ~240x cheaper at
        # len(seen)=586 and flat in it, and is bit-identical (each index is touched exactly once).
        if self.seen:
            idx = torch.tensor(sorted(self.seen))
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / self.penalty, v * self.penalty)
        logits = logits / TEMPERATURE
        if TOP_K and TOP_K < logits.numel():
            kth = torch.topk(logits, TOP_K).values[-1]
            logits[logits < kth] = float("-inf")
        if TOP_P < 1.0:
            sl, si = torch.sort(logits, descending=True)
            drop = torch.softmax(sl, dim=-1).cumsum(dim=-1) > TOP_P
            drop[1:] = drop[:-1].clone()
            drop[0] = False  # always keep the top-1
            logits[si[drop]] = float("-inf")
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=self.gen))

    def advance(self, lat, step):
        """Consume this step's latent: sample the next code and stage the next embedding.
        Returns False once the request has hit its stop token / the model's cap."""
        if step == 0:
            # START_AUDIO's latent predicts code_0; it is not itself a vocoder frame.
            self.codes.append(self.sample(lat))
        else:
            self.latents.append(lat)  # vocoder latent for the code just fed
            self.seen.add(self.codes[-1])
            nxt = self.sample(lat)
            if nxt == self.stop_audio or len(self.codes) >= GPT_MAX_AUDIO:
                self.done = True
                return False
            self.codes.append(nxt)
        j = len(self.codes) - 1  # the j-th generated code is fed at mel_pos[j+1]
        self.next_emb = (self.heads["mel_emb"][self.codes[-1]] + self.heads["mel_pos"][j + 1]).view(1, 1, -1)
        self.pos += 1
        return True

    def gpt_latents(self):
        return torch.cat(self.latents, dim=1) if self.latents else torch.zeros(1, 0, 1024)


def load_inputs(works):
    """Per-request Block-1/2 inputs. SPMD batches them, so they must share a shape; that holds
    when the requests share one reference clip (only the text differs)."""
    mels = [torch.load(os.path.join(w, "cond_mel_in.pt")).float() for w in works]
    lms = [torch.load(os.path.join(w, "speaker_logmel.pt")).float() for w in works]
    return mels, lms


# ---------------------------------------------------------------------------------------
# Flavor 1: SPMD — one instance on the whole mesh, requests sharded on dim 0
# ---------------------------------------------------------------------------------------
def run_spmd(mesh, works, heads, args):
    N = len(works)
    shard = ttnn.shard_tensor_to_mesh_mapper(mesh, 0)
    comp = ttnn.concat_mesh_to_tensor_composer(mesh, 0)

    slots = [RequestState(i, works[i], heads, 0 if args.same_seed else i) for i in range(N)]
    mels, lms = load_inputs(works)
    for m, lm in zip(mels, lms):
        assert m.shape == mels[0].shape and lm.shape == lms[0].shape, (
            "SPMD batches Blocks 1/2, so every request's conditioning mel must have the same "
            "length — use one reference clip for all requests (see the module docstring)"
        )

    t0 = time.time()
    spk = block2_speaker_embedding(mesh, torch.cat(lms, 0), shard, comp)  # [N,512,1]
    cond = block1_gpt_cond_latent(mesh, torch.cat(mels, 0), shard, comp)  # [N,32,1024]
    print(
        f"[M] Blocks 1+2 batched on {N} chips in {time.time() - t0:.1f}s  "
        f"spk {tuple(spk.shape)} cond {tuple(cond.shape)}"
    )

    # Right-pad the prompts to a common P; each request still decodes from its own P_i.
    Pmax = max(s.P for s in slots)
    prefix = torch.zeros(N, Pmax, 1024)
    for i, s in enumerate(slots):
        prefix[i, : s.P, :] = s.prefix[0]
        prefix[i, :LATENTS, :] = cond[i]  # use the TT-computed conditioning in the prompt
    print(f"[M] prompts P={[s.P for s in slots][:8]}{'...' if N > 8 else ''} -> padded Pmax={Pmax}")

    t0 = time.time()
    dec = TTNNGPTTracedDecoder(
        mesh,
        preprocess_gpt_parameters(mesh, dtype=ttnn.bfloat16),
        max_seq=Pmax + 1 + GPT_MAX_AUDIO,
        batch=N,
        data_mapper=shard,
    )
    dec.reset_caches()
    dec.prefill(prefix.contiguous())
    dec.capture()  # one trace for all N chips
    print(f"[M] GPT built + prefilled + traced in {time.time() - t0:.1f}s (max_seq={dec.max_seq})")

    live = list(slots)
    t0 = time.time()
    steps = 0
    for step in range(args.max_new + 1):
        if not live:
            break
        emb = torch.cat([s.next_emb for s in slots], 0)  # [N,1,1024]; finished slots are inert
        emb_dev = ttnn.from_torch(
            emb.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard
        )
        lat = ttnn.to_torch(dec.step_device(emb_dev, [s.pos for s in slots]), mesh_composer=comp).float()
        steps += 1
        live = [s for s in live if s.advance(lat[s.idx : s.idx + 1], step)]
        if step % 50 == 0:
            print(
                f"[M] step {step:4d}  live={len(live):2d}  codes(min/max)="
                f"{min(len(s.codes) for s in slots)}/{max(len(s.codes) for s in slots)}"
            )
    t_dec = time.time() - t0
    report_decode(slots, steps, t_dec, N)
    return slots, spk


# ---------------------------------------------------------------------------------------
# Flavor 2: replicas — N independent instances on 1x1 submeshes
# ---------------------------------------------------------------------------------------
class Replica:
    """One request pinned to one chip: its own submesh, model instance and decode state."""

    def __init__(self, state, dev, mel, logmel):
        self.s = state
        self.dev = dev
        self.mel = mel
        self.logmel = logmel
        self.pending = None

    def build(self):
        self.spk = block2_speaker_embedding(self.dev, self.logmel)  # [1,512,1]
        cond = block1_gpt_cond_latent(self.dev, self.mel)  # [1,32,1024]
        prefix = self.s.prefix.clone()
        prefix[:, :LATENTS, :] = cond
        self.dec = TTNNGPTTracedDecoder(
            self.dev,
            preprocess_gpt_parameters(self.dev, dtype=ttnn.bfloat16),
            max_seq=self.s.P + 1 + GPT_MAX_AUDIO,
        )
        self.dec.reset_caches()
        self.dec.prefill(prefix.contiguous())
        self.dec.capture()  # compile + capture the per-step decode graph (after prefill)
        return self

    def issue(self):
        """Copy this request's next embedding to its chip and replay the decode trace
        (non-blocking, so the next replica can be issued while this chip computes)."""
        emb_dev = ttnn.from_torch(
            self.s.next_emb.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev
        )
        self.pending = self.dec.step_device(emb_dev, self.s.pos)

    def collect(self, step):
        lat = ttnn.to_torch(self.pending).float()  # syncs this chip
        self.pending = None
        return self.s.advance(lat, step)


def run_replicas(mesh, works, heads, args):
    N = len(works)
    subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(subs) >= N, f"mesh gave {len(subs)} 1x1 submeshes, need {N}"
    for sm in subs[:N]:
        sm.enable_program_cache()

    mels, lms = load_inputs(works)
    t0 = time.time()
    reps = []
    for i in range(N):
        st = RequestState(i, works[i], heads, 0 if args.same_seed else i)
        reps.append(Replica(st, subs[i], mels[i], lms[i]).build())
        print(f"[M] replica {i:2d} built  P={st.P}  max_seq={reps[-1].dec.max_seq}  work={works[i]}")
    print(f"[M] all {N} replicas built in {time.time() - t0:.1f}s")

    slots = [r.s for r in reps]
    live = list(reps)
    t0 = time.time()
    steps = 0
    for step in range(args.max_new + 1):
        if not live:
            break
        for r in live:  # issue every live chip first: they compute concurrently
            r.issue()
        live = [r for r in live if r.collect(step)]
        steps += 1
        if step % 50 == 0:
            print(
                f"[M] step {step:4d}  live={len(live):2d}  codes(min/max)="
                f"{min(len(s.codes) for s in slots)}/{max(len(s.codes) for s in slots)}"
            )
    t_dec = time.time() - t0
    report_decode(slots, steps, t_dec, N)
    return slots, torch.cat([r.spk for r in reps], 0), reps


def report_decode(slots, steps, t_dec, N):
    tot = sum(len(s.codes) for s in slots)
    print(
        f"[M] decode done: {steps} steps, {t_dec:.1f}s, {tot} tokens over {N} requests "
        f"({1000 * t_dec / max(steps, 1):.1f} ms/step, {tot / max(t_dec, 1e-9):.1f} tok/s aggregate)"
    )
    for s in slots:
        print(f"[M] request {s.idx:2d}: {len(s.codes):4d} codes  first8={s.codes[:8]}")


def identity_check(slots, N):
    """With one --work dir and --same-seed every chip runs the identical request, so all
    outputs must match. Catches per-chip weight/state/mapping mistakes that PCC would miss."""
    ref = slots[0]
    bad = [s.idx for s in slots[1:] if s.codes != ref.codes]
    if bad:
        print(f"[M] ⚠ IDENTITY CHECK FAILED: chips {bad} produced different codes than chip 0")
        return False
    print(f"[M] identity check PASSED: all {N} chips produced identical codes ({len(ref.codes)})")
    worst = 0.0
    for s in slots[1:]:
        if len(s.latents) == len(ref.latents) and s.latents:
            worst = max(worst, (s.gpt_latents() - ref.gpt_latents()).abs().max().item())
    print(f"[M] max |latent difference| across chips: {worst:.3e}")
    return worst == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", action="append", required=True, help="phase-A work dir (repeat for N requests)")
    ap.add_argument("--replicas", type=int, default=32, help="number of chips / concurrent requests")
    ap.add_argument("--flavor", choices=["spmd", "replicas"], default="spmd")
    ap.add_argument("--max-new", type=int, default=GPT_MAX_AUDIO)
    ap.add_argument("--out", default=None, help="where to write per-request outputs (default: first work dir)")
    ap.add_argument(
        "--same-seed",
        action="store_true",
        help="give every request the same sampling seed; with a single --work dir this makes all "
        "chips run the identical request, so their outputs must match exactly (mesh sanity check)",
    )
    ap.add_argument("--skip-vocoder", action="store_true")
    args = ap.parse_args()

    N = args.replicas
    works = args.work if len(args.work) > 1 else args.work * N
    assert len(works) == N, f"got {len(args.work)} work dirs for {N} replicas (pass 1 or {N})"
    out_dir = args.out or works[0]
    replicated = len(set(works)) == 1

    heads = load_gen_head()  # mel_emb / mel_pos / mel_head_{w,b}: shared, host-side

    print(f"[M] opening 1x{N} mesh (flavor={args.flavor})")
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, N),
        l1_small_size=65536,  # conv halo config lives in L1_SMALL; the fp32 vocoder needs 64K (BUG-2/3)
        trace_region_size=60_000_000,  # the traced decode step
    )
    reps = None
    try:
        print(f"[M] mesh {mesh.shape} devices={mesh.get_num_devices()}")
        if args.flavor == "spmd":
            mesh.enable_program_cache()
            slots, spk = run_spmd(mesh, works, heads, args)
        else:
            slots, spk, reps = run_replicas(mesh, works, heads, args)

        ok = True
        if replicated and args.same_seed:
            ok = identity_check(slots, N)

        for s in slots:
            torch.save(s.gpt_latents(), os.path.join(out_dir, f"gpt_latents_tt_r{s.idx}.pt"))
        torch.save(spk, os.path.join(out_dir, "speaker_embedding_tt_mesh.pt"))

        if not args.skip_vocoder:
            t0 = time.time()
            lens = [s.gpt_latents().shape[1] for s in slots]
            if args.flavor == "spmd":
                # Pad to a common L for the batched call, then trim each waveform back.
                Lmax = max(lens)
                lat = torch.zeros(N, Lmax, 1024)
                for i, s in enumerate(slots):
                    if lens[i]:
                        lat[i, : lens[i], :] = s.gpt_latents()[0]
                wav = block4_vocoder(
                    mesh,
                    lat,
                    spk,
                    ttnn.shard_tensor_to_mesh_mapper(mesh, 0),
                    ttnn.concat_mesh_to_tensor_composer(mesh, 0),
                )
                for i, s in enumerate(slots):
                    n = int(wav.shape[-1] * lens[i] / Lmax) if Lmax else 0
                    torch.save(wav[i : i + 1, :, :n], os.path.join(out_dir, f"vocoder_wav_tt_r{s.idx}.pt"))
                print(
                    f"[M] vocoder: {N} waveforms in {time.time() - t0:.1f}s "
                    f"(padded to Lmax={Lmax}, trimmed per request)"
                )
            else:
                for r in reps:
                    if r.s.latents:
                        w = block4_vocoder(r.dev, r.s.gpt_latents(), r.spk)
                        torch.save(w, os.path.join(out_dir, f"vocoder_wav_tt_r{r.s.idx}.pt"))
                print(f"[M] vocoder: {N} waveforms in {time.time() - t0:.1f}s")

        print(f"[M] wrote per-request outputs to {out_dir}")
        if not ok:
            raise SystemExit(1)
    finally:
        # A mesh with carved submeshes shares its command queue with them; drain it first or
        # close throws "cq is in use by child submesh".
        mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

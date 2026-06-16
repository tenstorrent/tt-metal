# Master execution plan (audio wins + video) — 2026-06-14

Tracks everything requested. Branch with all audio work: `smarton/optimizer/ltx-audio-kernel`
(16 commits off `ltx-perf`, PCC 0.99379 throughout). Nothing merged to ltx-perf yet.

## Confirmed audio numbers (real, measured)
- E2E (bh 4x8sp1tp0_ring, gen#1 steady-state): **8.7s → 8.2s**; audio-decode stage **1.36s → 0.9s**.
- Audio-only test wall: ~704s (cold device graph-assembly 311s + CPU PCC oracle + warm 0.85s).
- Submesh sweep (warm decode / cold-assembly): 4x8 842ms/496s · 2x4 688ms/141s · **1x4 672ms/5.6s**.
- `LTX_AUDIO_SUBMESH=1x4` (gated, committed 6182a3b): warm 674ms, cold ~5.6s, PCC 0.99379.

## TASKS (status)
1. [ME] **Push good work to ltx-perf** — PR from ltx-audio-kernel → ltx-perf, before/after E2E
   timings in title+description. (Supersedes/extends PR #46789.) NOT a direct push to the shared
   ltx-perf branch — PR is the correct path.
2. [AGENT-audio] **Submesh sweep 1x2 (2-chip) + 1x1 (1-chip)** for audio, on top of 1x4. Land the
   FASTEST config to ltx-perf (PCC-gated). Default-on if the ttnn cq-close limitation allows for
   that submesh size; else keep env-gated + document.
3. [AGENT-audio] **~/audio-perf.md** — fine-grained profiling + lessons. MUST answer: *why* are
   smaller submeshes faster (exact mechanism), and *which* profiling numbers show it (cold-assembly
   311s→5.6s, cross-chip neighbor_pad halo barriers ∝ T-shard count, warm decode per-shard).
   Write AFTER the 1x2/1x1 sweep so it has the full curve.
4. [ME] **~/audio-scaling.md** — separate PLAN to make audio scale correctly and finish << 600ms.
   DO NOT IMPLEMENT. (Why audio doesn't scale today; what a scalable design looks like.)
5. [AGENT-video] **Execute ~/LTX_PERF_PLAN.md** — DiT/video optimization (baseline+profile →
   L1 bfp8_b+LoFi quant → L2 step-cut 10→7 → L4 sparse attn). Big, separate, PCC/quality-gated.
   Fill its §3 baseline slots first from a clean per-op profile (tt-npe + counters now available).
6. [ME] this file.

## Ordering / coordination
- (1) push first (locks the audio wins). (2)+(3) audio agent next. (5) video agent — biggest, runs
  on its own branch off the pushed result; serializes on the broker with (2).
- Multi-tenant: device only via broker, never reset/kill-foreign, SIGTERM-only own orphans.
- Disconnects kill background agents — relaunch from last commit / PLAN_PROGRESS.md if so.

## Done so far (this whole effort)
mel-VAE Conv3d blocking 3.3x · polyphase slice cuts · audio_only fast path · warm dev harness ·
BWE/VOC trace gated off (were regressions) · process_ops_logs 5x · tt-npe built · tt-buddy PR #8 ·
paradox resolved (vocoder gap is device-side cross-chip halo, not host dispatch).

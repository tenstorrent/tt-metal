# Autoresearch: ACE-Step v1.5 base — TTNN bring-up on Blackhole p150

## Objective
Bring up **ACE-Step/acestep-v15-base** (music-generation DiT) on a single Blackhole **p150**
card in TTNN, **methodically, module-by-module**, maximizing reuse of the **TTTv2** module
library (`models/common/modules/`). Each module: **write class → write PCC test → verify PCC
≥ threshold → move to next**. Never write from scratch what TTTv2 already provides.

Work lives in: `models/experimental/acestep/`
- `tt/`      — TTNN module implementations (only when TTTv2 is insufficient)
- `tests/pcc/` — one PCC test per module (template: BGE-M3 `tests/pcc/*.py`)
- `tests/test_utils.py` — shared helpers (already written)
- `reference/` — thin wrappers to pull HF reference submodules for PCC comparison

## The model (from HF config.json + modeling_acestep_v15_base.py)
ACE-Step v1.5 base = a **Qwen3-architecture DiT** + encoders + VAE. Key config:
- hidden_size=2048, intermediate_size=6144, num_hidden_layers=24
- num_attention_heads=16, num_key_value_heads=8 (GQA), head_dim=128
- hidden_act=silu, rms_norm_eps=1e-6, rope_theta=1e6, attention_bias=false
- layer_types alternate **sliding_attention (window=128) / full_attention**
- q_norm & k_norm = Qwen3RMSNorm on head_dim (per-head qk-norm)
- text_hidden_dim=1024, vocab_size=64003

**Building blocks (nn.Module classes in the HF file):**
1. `Qwen3MLP` — SwiGLU (gate/up/down), silu. → maps to TTTv2 **MLP1D**.
2. `Qwen3RMSNorm` — RMSNorm. → maps to TTTv2 **RMSNorm1D** (also used for per-head q/k norm).
3. `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` — RoPE θ=1e6. → TTTv2 **RotarySetup1D**.
4. `AceStepAttention` — GQA q/k/v/o proj, q_norm/k_norm, RoPE, sliding OR full, **also cross-attn
   mode** (no RoPE, kv from encoder_hidden_states). → start from TTTv2 **Attention1D**; extend
   for qk-norm + cross-attention + sliding window as needed.
5. `TimestepEmbedding` — sinusoidal + 2 linears + SiLU + `time_proj` → 6× modulation. Custom small.
6. `AceStepEncoderLayer` — pre-norm self-attn + MLP (bidirectional, no cache). Used by lyric encoder.
7. `AceStepDiTLayer` — **AdaLN** (scale_shift_table + temb → 6 chunks: shift/scale/gate for
   self-attn and MLP) + gated residual + self-attn + optional cross-attn + MLP. The core block.
8. `AceStepLyricEncoder` — Linear(text_hidden_dim→hidden) + N encoder layers + final norm.
9. Timbre encoder, attention pooler, FSQ (ResidualFSQ), audio decoder, 1D VAE — later stages.

## Bring-up order (do in THIS order — simplest first, build up)
1. **RMSNorm** (`RMSNorm1D`) — reference: `Qwen3RMSNorm`. Smallest, validates harness.
2. **MLP** (`MLP1D`) — reference: `Qwen3MLP` (SwiGLU). Confirms weight adapter + PCC flow.
3. **RoPE** (`RotarySetup1D`) — reference: `Qwen3RotaryEmbedding` + `apply_rotary_pos_emb`.
4. **Attention self (full)** — reference: `AceStepAttention` (full_attention layer, self, w/ qk-norm).
5. **Attention self (sliding window=128)** — same class, sliding layer_type.
6. **Attention cross** — `AceStepAttention` cross mode (kv from encoder, no RoPE).
7. **TimestepEmbedding** — custom small module (sinusoidal + MLP + 6× proj).
8. **AceStepEncoderLayer** — compose norm+attn+mlp.
9. **AceStepDiTLayer** — AdaLN modulation + self + cross + MLP (the payoff block).
10. **AceStepLyricEncoder** — stack of encoder layers.
11. Later: timbre encoder, attention pooler, FSQ, audio decoder / 1D VAE, full DiT stack, pipeline.

## Reuse rules (STRICT — do not violate)
- **TTTv2 first.** Import from `models/common/modules/` (MLP1D, RMSNorm1D, Attention1D,
  RotarySetup1D, Embedding1D, etc.). Read `models/common/modules/README.md` (the contract).
- If a TTTv2 module *almost* fits, use its **`from_config(<Name>Config)`** override path before
  writing anything new (README "Power Users" section).
- Only write a new class in `tt/` when TTTv2 genuinely cannot express the op (e.g. AdaLN
  modulation, timestep embedding, cross-attention kv-from-encoder, FSQ). When you do, **follow
  the TTTv2 contract**: `LightweightModule` subclass, `<Name>Config` dataclass, simple
  constructor + `from_config`, straight-line `forward` (no static if-else). Model file style on
  **Phi4** (`models/common/models/phi4/model.py`); file/test layout on **BGE-M3**
  (`models/demos/wormhole/bge_m3/`).
- Reference for PCC = the **real HF module** (trust_remote_code, weights from local HF cache
  `ACE-Step/acestep-v15-base`). Use real config dims (`tests/test_utils.py`). Do NOT invent shapes.

## Metrics
- **Primary**: `modules_passing` (count of PCC tests passing in `tests/pcc/`) — **higher is better**.
  This is the bring-up progress signal. Grows as each module is validated.
- **Secondary**: `min_pcc` (worst PCC across passing modules — watch for silent degradation),
  `avg_pcc`, `tests_total`, `tests_failing`, `suite_seconds`.

## How to Run
`./.auto/measure.sh` — runs `pytest models/experimental/acestep/tests/pcc/ -q`, counts passes,
extracts PCC values from output, emits `METRIC name=value` lines. With zero tests, baseline=0.

## PCC thresholds
- Norm/MLP/elementwise: **≥ 0.999**
- Attention / RoPE / composed blocks: **≥ 0.99**
- Full DiT layer / multi-layer stacks: **≥ 0.98** (accumulated bf16 error)
Never lower a threshold to force a pass — that is cheating. If PCC is low, fix the impl
(dtype, layout, weight transpose, norm eps, RoPE application order).

## Files in Scope
- `models/experimental/acestep/**` — all new code + tests. Free to modify.

## Off Limits
- `models/common/modules/**` (TTTv2 library — reuse, don't modify).
- `models/tt_dit/**`, `models/common/models/phi4/**`, `models/demos/wormhole/bge_m3/**`
  — references only, read don't edit.
- Any build files, submodules, CI.

## Constraints
- Single Blackhole **p150**, single-device (mesh 1×1). Use `require_single_device`.
- No new pip deps (transformers, torch, vector_quantize_pytorch already present).
- Each new module MUST have a passing PCC test before it counts as done.
- Do not overfit / cheat: real HF reference, real config dims, real thresholds.
- Keep tests fast where possible (small batch, representative seq lens).

## What's Been Tried (updated through Module 9)
**Strategy validated: the entire ACE-Step DiT backbone = 3 custom TTTv2-pattern classes +
pure TTTv2 reuse + one ttnn RoPE op.** All modules PCC-pass vs the genuine HF reference.

Pure TTTv2 reuse (ZERO custom code):
- M1 RMSNorm  -> RMSNorm1D (eps 1e-6 via from_config; simple ctor defaults 1e-5). PCC 0.9999.
- M2 MLP      -> MLP1D(w1=gate,w2=down,w3=up) SiLU. HF Linear [out,in] .transpose to [in,out]. 0.9999.
- M3 RoPE     -> ttnn.experimental.rotary_embedding_hf (HF rotate_half convention, NOT the
                Meta-permute rotary_embedding_llama). cos/sin [1,1,seq,head_dim] TILE. 0.9999.

Custom classes in tt/ (follow TTTv2 contract; reuse ttnn ops + RMSNorm1D/MLP1D inside):
- tt/attention.py  AceStepAttention — bidirectional GQA (16/8), per-head qk-norm via RMSNorm1D,
    rotary_embedding_hf, ttnn SDPA is_causal=False. ONE class covers 3 modes: full-self,
    sliding-self (window=128 via additive 4D mask), cross (kv from encoder, no RoPE). M4/5/6.
- tt/timestep_embedding.py  TimestepEmbedding — host sinusoidal(256) + device Linears +
    time_proj->6x modulation. temb & timestep_proj both 0.999. M7.
- tt/encoder_layer.py  AceStepEncoderLayer — pre-norm self-attn+MLP+residuals (full+sliding). M8.
- tt/dit_layer.py  AceStepDiTLayer — AdaLN (scale_shift_table+temb).chunk(6) gated residuals +
    self + cross + MLP. THE core generative block. PCC>=0.98. M9.

Key gotchas learned:
- Reference loading: `uv pip install vector_quantize_pytorch` INTO the venv (not ~/.local);
  apg_guidance.py is absent from snapshot -> stubbed in reference/hf_reference.py; MUST set
  `cfg._attn_implementation='eager'` or HF attention crashes KeyError:None.
- Commit flow: pre-commit black reformats files and aborts the FIRST commit; run `git add -A &&
  git commit` a SECOND time. NEVER use --amend with hooks. (Local autoresearch git automation
  can't reach the remote repo — commit manually on remote; ignore the local 'git add failed'.)
- Harness: autoresearch runs LOCALLY (mac); repo is REMOTE. measure runs via
  `ssh -o BatchMode=yes sjc-snva-tp100 'cd <repo> && ./.auto/measure.sh'`. Edit with ssh_*.

## Remaining (bring-up order)
- M10 AceStepLyricEncoder: Linear(text_hidden_dim=1024->hidden) + N=8 AceStepEncoderLayer + norm.
- M11 stack of DiT layers (24) — verify multi-layer accumulation stays PCC>=0.97.
- M12+ timbre encoder, attention pooler, FSQ (ResidualFSQ), 1D VAE audio decoder
  (biggest from-scratch pieces — check models/tt_dit/models/audio_vae + layers/audio_ops.py FIRST).
- Full DiT model + pipeline wiring.

## Key gotchas to remember
- ACE-Step uses **per-head q/k RMSNorm** (Qwen3-style) — Attention1D may need qk-norm wired in.
- **AdaLN**: `(scale_shift_table + temb).chunk(6)` → (shift,scale,gate)×2; residual is *gated*.
- Cross-attention: **no RoPE**, kv comes from `encoder_hidden_states`, not causal.
- Sliding layers use window=128; full layers are global. Alternating per `layer_types`.
- Weights: HF stores Linear as [out,in]; TTNN matmul wants [in,out] → transpose on load
  (see BGE-M3 `make_lazy_weight(... .transpose(-1,-2))`).
- RoPE θ = 1,000,000 (not 10k).

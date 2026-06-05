# Distributed RMSNorm usage in the LTX-2.3 transformer block (text→video+audio, AV)

Scope: **LTX-2.3 distilled text-to-video+audio (AV, `has_audio=True`)**, 1080p / 24fps /
~6s (`145` frames, `1088×1920`), on **Blackhole 2×4** and **Blackhole 4×8 (Galaxy, ring)**.

Source files:
- Block: `models/tt_dit/models/transformers/ltx/transformer_ltx.py` (`LTXTransformerBlock`)
- Attention: `models/tt_dit/models/transformers/ltx/attention_ltx.py` (`LTXAttention`)
- Norm op: `models/tt_dit/layers/normalization.py` (`DistributedRMSNorm`)
- Pipeline / config: `models/tt_dit/pipelines/ltx/pipeline_ltx.py`, `pipeline_ltx_distilled.py`

---

## 0. Config that fixes every number below

| Quantity | Value | Source |
|---|---|---|
| `num_layers` | 48 | `pipeline_ltx.py:202` |
| `video_dim` / `inner_dim` | 32·128 = **4096** (32 heads, head_dim 128) | `transformer_ltx.py:483` |
| `audio_dim` / `audio_inner_dim` | 32·64 = **2048** (32 heads, head_dim 64) | `transformer_ltx.py:494` |
| `cross_attention_dim` (video text) | 4096 | `pipeline_ltx.py:203` |
| `audio_cross_attention_dim` (audio text) | 2048 | `transformer_ltx.py:469` |
| `cross_attention_adaln` | **True** (9-param adaLN; 22B-distilled ckpt) | `pipeline_ltx.py:484-488` |
| `has_audio` | **True** (AV) | `transformer_ltx.py:111` |
| `skip_cross_attn` (A↔V) | **False** → A↔V cross-attn runs | distilled forward default |
| `apply_gated_attention` | ckpt-detected; off for distilled-1.1 | `pipeline_ltx.py:489` |
| `is_fsdp` | False | `pipeline_ltx.py:398,406` |
| `norm_eps` | 1e-6 | `transformer_ltx.py:35` |

Parallelism (`pipeline_ltx.py:391-446`; distilled delegates here, `pipeline_ltx_distilled.py:34-36`).
`DistributedRMSNorm` shards the **reduction (feature) dim** on the **TP axis**; activation
rows are **SP-sharded** on the sequence dim.

| HW | mesh | SP (axis) | TP (axis) | topology | `video_dim/TP` | `audio_dim/TP` |
|---|---|---|---|---|---|---|
| **BH 2×4** | (2,4) | **4**, axis 1 | **2**, axis 0 | Linear | 2048 | 1024 |
| **BH 4×8** | (4,8) | **8**, axis 1 | **4**, axis 0 | Ring | 1024 | 512 |

Sequence lengths (`B=1`; no CFG). `N` padded to `TILE·SP = 32·SP`; `N_local = N/SP`:

| Tensor | `N_real` | BH 2×4 `N` / `N_local` | BH 4×8 `N` / `N_local` |
|---|---|---|---|
| Video, Stage 1 (half-res 544×960) | 19·17·30 = 9 690 | 9 728 / **2 432** | 9 728 / **1 216** |
| Video, Stage 2 (1080p 1088×1920) | 19·34·60 = 38 760 | 38 784 / **9 696** | 38 912 / **4 864** |
| Audio (both stages) | 151 | 256 / **64** | 256 / **32** |

Audio length: `dur=145/24=6.04s`, `25 latents/s` → `round(151.04)=151`
(`patchifiers.py:52-56`). Video latent grid: `frames=(145-1)//8+1=19`, spatial `//32`
(`pipeline_ltx.py:69-93`). Text-prompt length (video & audio): **L = 1024** (Gemma,
`encoders/gemma/encoder_pair.py:45`), replicated across SP.

---

## 1. Which DistributedRMSNorm modules run per block (AV)

**18 module instances → 20 invocations per block** (`norm3` and `audio_norm3` are each
called twice: once as the A↔V cross-attn pre-norm, once as the FFN pre-norm). All
`norm_eps=1e-6`, `bias=False`. ×48 layers = **960 invocations / denoise pass**.

### Video path (`embedding_dim = 4096`, head_dim 128)

| # | Module | Where (def / call) | Role | Affine | Trailing addcmul | RoPE | create_heads |
|---|---|---|---|---|---|---|---|
| 1 | `norm1` | `:75` / `:257` | pre video self-attn | No | **Yes** `:258` | n/a | no |
| 2 | `attn1.norm_q` | `attn:79` / `:394` | self-attn Q | Yes | no | **Yes** (sep. `:426`) | **fused** |
| 3 | `attn1.norm_k` | `attn:80` / `:395` | self-attn K | Yes | no | **Yes** (sep. `:429`) | **fused** |
| 4 | `norm2` | `:77` / `:272` | pre video text cross-attn | No | **Yes** `:272` | n/a | no |
| 5 | `attn2.norm_q` | `attn:79` / `:394` | text-cross Q (video) | Yes | no | No | **fused** |
| 6 | `attn2.norm_k` | `attn:80` / `:395` | text-cross K (text, L) | Yes | no | No | **fused** |
| 7 | `norm3` (call A) | `:85` / `:366` | pre A↔V cross-attn (video) | No | **Yes** `:370,:394` | n/a | no |
| 8 | `norm3` (call B) | `:85` / `:411` | pre video FFN | No | **Yes** `:412` | n/a | no |

### Audio path (`embedding_dim = 2048`, head_dim 64)

| # | Module | Where | Role | Affine | Trailing addcmul | RoPE | create_heads |
|---|---|---|---|---|---|---|---|
| 9 | `audio_norm1` | `:112` / `:323` | pre audio self-attn | No | **Yes** `:324` | n/a | no |
| 10 | `audio_attn1.norm_q` | `attn:79` / `:394` | self-attn Q | Yes | no | **Yes** (sep.) | **fused** |
| 11 | `audio_attn1.norm_k` | `attn:80` / `:395` | self-attn K | Yes | no | **Yes** (sep.) | **fused** |
| 12 | `audio_norm2` | `:114` / `:339` | pre audio text cross-attn | No | **Yes** `:339` | n/a | no |
| 13 | `audio_attn2.norm_q` | `attn:79` / `:394` | text-cross Q (audio) | Yes | no | No | **fused** |
| 14 | `audio_attn2.norm_k` | `attn:80` / `:395` | text-cross K (text, L) | Yes | no | No | **fused** |
| 15 | `audio_norm3` (call A) | `:122` / `:367` | pre A↔V cross-attn (audio) | No | **Yes** `:371,:393` | n/a | no |
| 16 | `audio_norm3` (call B) | `:122` / `:431` | pre audio FFN | No | **Yes** `:432` | n/a | no |

### A↔V cross-attention path (`embedding_dim = 2048` = `audio_dim`, head_dim 64)

These two attentions are `dim=audio_dim` modules (`transformer_ltx.py:146-161`); their
QK norms are therefore 2048-wide even where Q/K originate from the 4096-wide stream
(`to_q`/`to_kv` project to 2048 first).

| # | Module | Where | Role | Affine | RoPE | create_heads |
|---|---|---|---|---|---|---|
| 17 | `audio_to_video_attn.norm_q` | `attn:79` / `:394` | A→V Q (from video) | Yes | **Yes** (video cross-PE `:384`) | **fused** |
| 18 | `audio_to_video_attn.norm_k` | `attn:80` / `:395` | A→V K (audio ctx, full) | Yes | **Yes** (audio cross-PE `:386`) | **fused** |
| 19 | `video_to_audio_attn.norm_q` | `attn:79` / `:394` | V→A Q (from audio) | Yes | **Yes** (audio cross-PE `:402`) | **fused** |
| 20 | `video_to_audio_attn.norm_k` | `attn:80` / `:395` | V→A K (video ctx, local) | Yes | **Yes** (video cross-PE `:404`) | **fused** |

Two usage patterns, same as T2V but doubled across modalities:
- **Block norms** (`norm1/2/3`, `audio_norm1/2/3`): no static affine; each immediately
  followed by a *separate* `ttnn.addcmul` adaLN modulation (shift+scale).
- **Attention QK norms** (all `norm_q/norm_k`): static affine weight folded into the op,
  fused with the head-split; RoPE applied as a *separate* op where present (self-attn and
  both A↔V cross-attns; not text cross-attn).

---

## 2. Settings per invocation

`DistributedRMSNorm.__init__` (`normalization.py:139-183`): common to all 20 —
`norm_eps=1e-6`, `bias=False` (asserted off), `mesh_axis = TP axis`,
`compute_kernel_config = HiFi4 + fp32_dest_acc + packer_l1_acc=False`.

| Setting | Block norms (`*norm1/2/3`) | Attention QK norms |
|---|---|---|
| `embedding_dim` | 4096 (video) / 2048 (audio) | 4096 (video attn) / 2048 (audio attn + both A↔V) |
| `norm_elementwise_affine` | **False** → `weight=None` | **True** → `weight` sharded on TP axis |
| forward `num_heads_per_device` | 1 | `n_local_heads` = `num_heads/TP` (16 on 2×4, 8 on 4×8) |
| forward `rope_*` | None | None passed (RoPE is a separate op) |

`forward` (`normalization.py:189-230`): `wan_fused_rmsnorm_pre_allgather` → TP all-gather
of partial stats (TP>1) → `wan_fused_rmsnorm_post_allgather(weight, num_heads_per_device,
rope_cos, rope_sin, transformation_mat, ...)`. The post-allgather op already accepts
`weight`, `num_heads_per_device`, **and** RoPE inputs — only the first two are used today.

---

## 3. RoPE + create_heads — what is fused, what isn't

Applies to QK norms only.
- **`create_heads` — already fused** everywhere via `num_heads_per_device` (the op emits
  `BHNE` directly, `attention_ltx.py:393-395`). The explicit `nlp_create_qkv_heads`
  (`:397-401`) is used only for V.
- **RoPE — NOT fused (opportunity).** Applied as a standalone
  `rotary_embedding_llama(q_BHNE, cos, sin, trans_mat)` (`attention_ltx.py:426-431`). The
  norm op already accepts `rope_cos/sin/trans_mat` but the block passes none. Fusible
  wherever RoPE is present:
  - self-attn `attn1` + `audio_attn1` (#2,3,10,11),
  - **both A↔V cross-attns** `audio_to_video_attn` + `video_to_audio_attn` (#17–20) — note
    these carry separate Q-side and K-side cross-PEs.
  - **Not** the text cross-attns (#5,6,13,14) — no RoPE there.

  **RoPE cos/sin are PER-HEAD (non-standard).** Standard llama RoPE broadcasts one
  `(1, 1, N, head_dim)` table across all heads. LTX-2 instead computes frequencies over the
  *full* `inner_dim` and reshapes to per-head slices (`precompute_freqs_cis` +
  `reshape_interleaved_to_bhnd`, `rope_ltx.py:100-104`, `pipeline_ltx.py:859-871`), so every
  head has a **distinct** frequency band → cos/sin carry a real head axis:
  `(1, num_heads, N, head_dim)`. The last dim is the **full** `head_dim` (not `head_dim/2`):
  each unique freq is `repeat_interleave(2)`-duplicated for the `(x0,x1)` interleaved pairs
  (`rope_ltx.py:114-115`). A fused norm+RoPE path must therefore consume a per-(head, token)
  cos/sin, not a per-token-only table. Exact shapes in §6.

---

## 4. addcmul ternary (shift "bias" + scale "weight") — what could be fused

Applies to the **block norms** (`norm1/2/3`, `audio_norm1/2/3`), which have no static
affine. Each is followed by a *separate* `ttnn.addcmul(shift, normed, scale_p1)`:

| Norm (call) | addcmul line | shift / scale operands |
|---|---|---|
| `norm1` | `:258` | `v_shift_sa` / `v_scale_sa_p1` |
| `norm2` | `:272` | `v_shift_ca` / `v_scale_ca_p1` |
| `norm3` (A, A↔V) | `:370`, `:394` | feeds A→V query (`v_ca_*`) **and** V→A kv (`a_ca_*`) |
| `norm3` (B, FFN) | `:412` | `v_shift_ff` / `v_scale_ff_p1` |
| `audio_norm1` | `:324` | `a_shift_sa` / `a_scale_sa_p1` |
| `audio_norm2` | `:339` | `a_shift_ca` / `a_scale_ca_p1` |
| `audio_norm3` (A, A↔V) | `:371`, `:393` | feeds A→V kv (`a_shift_a2v`) **and** V→A query (`a_shift_v2a`) |
| `audio_norm3` (B, FFN) | `:432` | `a_shift_ff` / `a_scale_ff_p1` |

`addcmul(input, t1, t2, value) = input + value·t1·t2`: the scalar **`value` is 1.0** in
every case (default; fused paths pass `scalar=1.0` explicitly). The **scale operand carries
`1+scale`** — the `+1` is pre-baked into the scale-shift tables at load
(`_prepare_torch_state`, `transformer_ltx.py:191-211`; A↔V tables use scale slots
`[0,2]`, `:205-206`), so `normed·(1+scale)+shift` collapses to one op. These standalone
addcmuls are fusion candidates into the post-allgather step as a **dynamic (per-token)
ternary affine** (distinct from the static `weight=` path).

**The addcmul shift/scale operands are PER-TOKEN activations, not per-channel weights.**
A static RMS affine is a `[1, D]` per-channel constant. Here the modulation comes from
`adaln_single(timestep)` (`transformer_ltx.py:770`); the timestep is per-sample
`(1,1,B,1)` (`:695`), reshaped to `video_mod_1BCD = (1, B, 9, inner_dim)`, TP-partitioned
on the feature dim (`:772-776`), then `ttnn.chunk(..., dim=2)` yields each shift/scale/gate
as **`(1, B, 1, D/TP)`** (`:250-254`). The op broadcasts this over the `N` token rows of
`normed (1, B, N, D/TP)`. So unlike a static norm weight, the addcmul operands are
runtime, per-step, batch-dim tensors fed as activations and broadcast across the sequence —
a fused norm+modulation path must accept a `(1, B, 1, D/TP)` ternary broadcast, distinct
from the static `weight=` arg. Exact shapes in §6.

QK norms have **no** trailing addcmul (their affine is the static RMS weight, already
fused). Other block addcmuls are fused elsewhere: self-attn output gate → `to_out`
(`_to_out_fused_addcmul`, `attention_ltx.py:243-309`); FFN gate → FFN on Ring
(`forward_fused_addcmul`, `transformer_ltx.py:296-304,413-421`).

---

## 5. Input shapes to each DistributedRMSNorm

Per-device local tensors `(1, 1, rows, feat_local)`. `feat_local = embedding_dim/TP`.
`rows`: `N_local` for SP-sharded streams; `L=1024` for text K; `audio_N=256` (full) for the
A→V audio context; `video_N_local` (local) for the V→A video context.

Numbers: `video_N_local` = 2432/1216 (S1) · 9696/4864 (S2) for 2×4/4×8; `audio_N_local` =
64/32; `audio_N` = 256; `L` = 1024.

### Video block norms `norm1/norm2/norm3` — `(1,1, video_N_local, 4096/TP)`
| HW | Stage 1 | Stage 2 (1080p) |
|---|---|---|
| BH 2×4 | `(1,1,2432,2048)` | `(1,1,9696,2048)` |
| BH 4×8 | `(1,1,1216,1024)` | `(1,1,4864,1024)` |

### Video self-attn QK (`attn1`) — `(1,1, video_N_local, 4096/TP)` → out `(1, 32/TP, video_N_local, 128)`
Same input table as above. e.g. BH 4×8 S2: in `(1,1,4864,1024)` → out `(1,8,4864,128)`.

### Video text-cross QK (`attn2`)
- `norm_q` (video): `(1,1, video_N_local, 4096/TP)` (table above).
- `norm_k` (text, L): `(1,1,1024,2048)` (2×4) / `(1,1,1024,1024)` (4×8).

### Audio block norms `audio_norm1/2/3` — `(1,1, audio_N_local, 2048/TP)`
| HW | both stages |
|---|---|
| BH 2×4 | `(1,1,64,1024)` |
| BH 4×8 | `(1,1,32,512)` |

### Audio self-attn QK (`audio_attn1`) — `(1,1, audio_N_local, 2048/TP)` → out `(1, 32/TP, audio_N_local, 64)`

### Audio text-cross QK (`audio_attn2`)
- `norm_q` (audio): `(1,1, audio_N_local, 2048/TP)`.
- `norm_k` (text, L): `(1,1,1024,1024)` (2×4) / `(1,1,1024,512)` (4×8).

### A→V cross (`audio_to_video_attn`, `embedding_dim=2048`, head_dim 64)
- `norm_q` (video query, `to_q` 4096→2048): `(1,1, video_N_local, 2048/TP)`.
- `norm_k` (audio context, SP-gathered to full `audio_N=256`): `(1,1,256,2048/TP)` →
  `(1,1,256,1024)` (2×4) / `(1,1,256,512)` (4×8).

### V→A cross (`video_to_audio_attn`, `embedding_dim=2048`, head_dim 64)
- `norm_q` (audio query): `(1,1, audio_N_local, 2048/TP)`.
- `norm_k` (video context, local shard before SDPA gather): `(1,1, video_N_local, 2048/TP)`.

Each op reduces over the **full** feature dim via the TP all-gather of partial stats
(`normalization.py:211-216`); `feat_local` is the shard each device holds.

---

## 6. Shapes of the fusible side-tensors (RoPE cos/sin & adaLN modulation)

These are the auxiliary operands a fused norm kernel would have to ingest. Both differ from
the "standard" assumptions, which is why they're called out here.

### 6a. RoPE cos/sin — **per-head**, last dim = full `head_dim`

Built by `precompute_freqs_cis` over the full inner dim → `reshape_interleaved_to_bhnd`
→ `(1, num_heads, N, head_dim)`, with each frequency `repeat_interleave(2)`-duplicated so
the last dim is the **whole** `head_dim` (interleaved `(x0,x1)` layout for
`rotary_embedding_llama`). `num_heads = 32` everywhere. Sharded `{sp_axis:2, tp_axis:1}`
(Q-side and self-attn) or TP-only on the head axis (A↔V K-side, sequence replicated post
all-gather).

| RoPE tensor (used by) | dim | head_dim | full shape `(1, H, N, hd)` | local `(1, H/TP, rows, hd)` — BH 2×4 / 4×8 |
|---|---|---|---|---|
| video self-attn (#2,3, `attn1`) | 4096 | 128 | `(1, 32, video_N, 128)` | `(1,16, video_N_local,128)` / `(1,8, video_N_local,128)` |
| audio self-attn (#10,11, `audio_attn1`) | 2048 | 64 | `(1, 32, audio_N, 64)` | `(1,16, audio_N_local,64)` / `(1,8, audio_N_local,64)` |
| A→V video-Q (#17) | 2048 | 64 | `(1, 32, video_N, 64)` | `(1,16, video_N_local,64)` / `(1,8, video_N_local,64)` |
| A→V audio-K (#18, TP-only) | 2048 | 64 | `(1, 32, audio_N, 64)` | `(1,16, 256, 64)` / `(1,8, 256, 64)` |
| V→A audio-Q (#19) | 2048 | 64 | `(1, 32, audio_N, 64)` | `(1,16, audio_N_local,64)` / `(1,8, audio_N_local,64)` |
| V→A video-K (#20, TP-only) | 2048 | 64 | `(1, 32, video_N, 64)` | `(1,16, video_N_full,64)` / `(1,8, video_N_full,64)` |

(`video_N_local` = 2432/1216 S1, 9696/4864 S2; `audio_N_local` = 64/32; `audio_N`=256.
Source: `pipeline_ltx.py:859-881` video, `:1024-1054` audio, `:1076-1140` A↔V cross-PE.
A↔V uses `dim=2048` for *both* sides so audio@t and video@t share rotary phase, hence the
64-wide head_dim even on the video Q/K.)

Note the video self-attn RoPE is `head_dim=128`, but the video Q/K that flow through the
A↔V cross-attns get a **64**-wide RoPE (those attentions are `dim=audio_dim=2048`).

### 6b. adaLN modulation (addcmul shift/scale/gate) — **per-token broadcast**, `(1, B, 1, D/TP)`

One modulation vector per sample/timestep, broadcast over all `N` tokens (constant along the
sequence; varies along the feature dim). `B=1`. Source: `transformer_ltx.py:249-254`
(video), `:315-320` (audio), `:360-364` (A↔V).

| Modulation operand group | D | shape `(1, B, 1, D/TP)` — BH 2×4 / 4×8 | feeds (addcmul) |
|---|---|---|---|
| video shift/scale/gate (×3 pairs: SA/CA/FF) | 4096 | `(1,1,1,2048)` / `(1,1,1,1024)` | `norm1/2/3` → `:258,:272,:412` |
| audio shift/scale/gate (×3 pairs) | 2048 | `(1,1,1,1024)` / `(1,1,1,512)` | `audio_norm1/2/3` → `:324,:339,:432` |
| A↔V video table (`scale_shift_table_a2v_ca_video`, 5 chunks) | 4096 | `(1,1,1,2048)` / `(1,1,1,1024)` | `norm3` → `:370,:394` |
| A↔V audio table (`scale_shift_table_a2v_ca_audio`, 5 chunks) | 2048 | `(1,1,1,1024)` / `(1,1,1,512)` | `audio_norm3` → `:371,:393` |

Contrast with the RoPE tables (§6a), which vary per **token and per head**; the modulation
varies per **channel** only and is identical across all tokens of a sample.

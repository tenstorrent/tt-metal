# tt_pipeline PORT_NOTES

Standalone port of the `tt_symbiote` `pipelined_pi05` streamed-denoise path into
`tt-metal_pipeline` (`models/experimental/pi0_5/tt/tt_pipeline/`). Self-contained sibling of
`tt/tt_bh_glx/`. ZERO `tt_symbiote` imports; `tt_bh_glx/` BYTE-UNCHANGED.

## references_read (CLAUDE.md tech-report gate)

- timestamp: 2026-06-24T19:33Z
- target tt-metal commit (`git -C . rev-parse HEAD`): `58672b47cfd304195798bcf34d44f5dbcbcf5189`
  (baked into `TT_METAL_COMMIT` in every new file, replacing the source pins
  `7d1b555a…` / `2475f8f0…` / `b0703a56…` / `b2af0cd6…`).
- source reference impls consulted (tt_symbiote):
  - `core/module.py`, `core/arch.py`, `core/run_config.py` (trace symbols only),
    `core/d2d_bridge.py`, `core/d2d_pipeline.py`, `core/d2d_transport.py`,
    `utils/device_management.py`
  - `models/pi05/modeling_pi05_{pcfg,common,bs,gemma,suffix}.py`
  - `models/pipelined_pi05/{denoise_block,denoise_pipeline}.py`
- target reference impls consulted (tt-metal_pipeline):
  - `tt/tt_bh_glx/transport.py` (parent SocketTransport -- subclassed, NOT edited)
  - `common/configs.py` (GemmaConfig / SuffixConfig / Pi0_5ModelConfig -- reused; fields verified)
  - `common/weight_loader.py` (categorized_weights / get_layer_weights -- key map verified)
  - `reference/torch_gemma.py` (AdaRMSGemmaBlock -- reused as the 18 reference blocks)
  - `reference/torch_suffix.py` (Pi0_5SuffixEmbedding -- reused as the reference suffix;
    `embed_timestep_adarms` is the host adarms fn)
  - `reference/torch_paligemma.py` (`model.norm.dense.{weight,bias}` final-mod source)
  - `reference/torch_denoise.py` (DenoisingModule.sample_actions / denoise_step golden)
- tech reports (read for this port): `ttnn/TTNN-model-bringup.md`, `LLMs/llms.md`,
  `FlashAttention/FlashAttention.md`, `tensor_layouts/tensor_layouts.md`,
  `tensor_sharding/tensor_sharding.md`, `data_formats/data_formats.md`,
  `AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md`,
  `MetalProfiler/metal-profiler.md`.

## Step-2 source-fact confirmations

- (a) PI0_MM_SWEEP_V2 table V2 ⊇ BASE: V2 = BASE + `(32,80)->(64,8)`. **V2 baked ON.**
- (b) `get_layer_weights(i, "action_expert")` returns `model.layers.{i}.<local>` keys; the
  reference `AdaRMSGemmaBlock(config, layer_weights, i)` reads layer-local keys
  (`input_layernorm.dense.weight`, `self_attn.q_proj.weight`, `mlp.gate_proj.weight`, ...).
  `weight_adapt.expert_reference_blocks` slices the per-layer sub-dict (strips prefix). VERIFIED.
- (c) host adarms fn is `Pi0_5SuffixEmbedding.embed_timestep_adarms`
  (sincos -> Linear -> silu -> Linear -> silu). **(prose copy-edit: NOT `embed_adarms_cond`.)**
- (d) `ttnn_matmul_decode` ABSENT from the denoise path (`grep -rn ttnn_matmul_decode tt/` empty).
  Not vendored.
- (e) `modeling_pi05_pcfg.py` read and vendored verbatim into `modeling/pcfg.py`.
- `_RefSuffix` G5 seam: the pi05 SUBCLASS `Pi0_5SuffixEmbedding.__init__` sets the
  `action_in/out` + `time_mlp_in/out` attrs UNCONDITIONALLY (un-prefixed keys for pi05_base),
  unlike the base class which only sets time_mlp_* in its `if not config.pi05` branch. So the
  target `Pi0_5SuffixEmbedding` IS reusable directly as the reference suffix for pi05 (it is
  NOT the base class). VERIFIED at `reference/torch_suffix.py:211-235`.
- Galaxy `transport._pair` hard-codes `_N_SOCK_CONN` (module default 2) via a module constant;
  `SplitSocketTransport._pair` builds the SocketConnection list directly with `num_connections=1`
  (open item #5 resolved: build directly).

## Env-flag bake decisions (plan §9)

- PI0_DENOISE_MM_TUNE -> ON; PI0_MM_SWEEP_V2 -> ON (V2 table); PI0_EXPERT_MM_LOFI -> OFF (HiFi;
  documented LoFi flip in `denoise_block._expert_lofi_ck`); PI0_DENOISE_MLP_WIDE -> OFF (wide
  variant dropped); PI0_SDPA_DENOISE_K_FORCE -> dropped (auto); PI05_VLM_KV_BF16 -> OFF
  (bfloat8_b; bf16 PCC-recovery flip in `gemma.init_static_kv` + `denoise_pipeline._kv_dtype`);
  PI05_PERF_AH -> dropped (real action_horizon); PI05_GLX_TRANSPORT -> dropped (direct sockets);
  PI05_SOCK_CONN -> baked 1 in SplitSocketTransport; LADDER_*/PI05_VLM_* -> baked deterministic
  (HiFi4, fp32_dest=False, packer_l1=True). `grep -rn "os.environ" tt/tt_pipeline/` is EMPTY.

## Phase-B recovery flips (gated on tracy + PCC re-validation)

1. 8-way layer partition `(3,3,3,3,2,2,1,1)` -- tunable.
2. LoFi expert matmul (`_expert_lofi_ck` returns the LoFi WormholeComputeKernelConfig).
3. KV bf16 (`_kv_dtype` -> bfloat16 + `init_static_kv` static dtype -> bfloat16).

## iter-3 ROOT CAUSE + FIX (multi-stage streamed PCC gap closed)

- timestamp: 2026-06-24T21:40Z; tt-metal commit `58672b47cfd304195798bcf34d44f5dbcbcf5189`.
- tech reports re-consulted: `FlashAttention/FlashAttention.md` (attn-mask semantics),
  `data_formats/data_formats.md` (bf16 magnitude vs PCC), `MetalProfiler/metal-profiler.md`
  (real-time profiler behaviour). Reference impls re-read: `reference/torch_paligemma.forward_expert`
  (mask=None), `reference/torch_gemma.{ada_rms_norm,GemmaAttention.forward}`.

### §4 finding (verified on device, action_horizon=50 -> suffix_len=64 -> m_tiles=2)
The tuned matmul table is DEAD CODE (`_denoise_tuned_pcfg` returns None for m_tiles!=1),
`_expert_lofi_ck` returns None (HiFi), so the streamed-expert compute is byte-identical to the
parent block except the (benign) SDPA-chunk choice. R0.5 oracle confirmed BLOCK_BENIGN.

### Drill-down evidence (1 Euler step, eager, 4 chips, concat-KV)
Tap PCC / rel_l2 / norm_ratio vs the torch golden:
- T0 x_t->bf16: bit-exact. T6 velocity-wrap socket: BIT-EXACT (lossless). T6b fp32 typecast:
  no recovery needed (== T5). => H-V / H-Hop streaming hypotheses REFUTED.
- T5 velocity (real rows): PCC 0.9915, **rel_l2 0.339, norm_ratio 1.304** (device velocity
  ~1.30x too LARGE in magnitude; PCC is scale-invariant so it hid this).
- T7 integrate (x_t + dt*v): PCC drops to 0.896; ACTIONS 0.9026.
- **Single-chip oracle R0.5 (parent static-KV AND expert concat-KV) reproduces IDENTICAL
  numbers: velocity 0.991 / norm_ratio 1.304 / INTEGRATED 0.9026.** => the multi-stage
  STREAMING contributes ZERO error; the gap is inherent to the single-chip expert compute.
- Single-block isolation: (offset=prefix_len, NO mask) PCC 0.9999 vs (offset=0, WITH the
  `-1e4` phantom mask) PCC 0.9936 (rel_l2 0.11/layer). => the **PHANTOM MASK is the divergence**,
  NOT the RoPE offset, NOT block-class, NOT precision, NOT streaming.

### Precision-vs-logic verdict: LOGIC (structured per-layer mask divergence, not bf16 drift).
The torch golden (`forward_expert`) runs with `attention_mask=None`; the iter-2 `-1e4` phantom
band on the pad-suffix KEY positions `[prefix_len+ah : prefix_len+suffix_len]` made the device
attention distribution diverge ~11% rel-L2 per layer, compounding over 18 layers into the
~1.30x velocity-magnitude error. PCC hid it; the magnitude-sensitive Euler integrate exposed it.

### Fix applied (Fix-Mask -- NOT in the plan's H-V/E/KV/Hop list; the drill localized a NEW cause)
`stage_denoise._build_phantom_mask_and_offset` now returns an ALL-ZEROS additive mask (no `-1e4`
band) -> golden parity. RoPE offset=prefix_len KEPT (drill-confirmed correct). One-line change,
smallest blast radius, no fp32 promotions. Fix-V/Fix-E/Fix-KV/Fix-Hop were NOT applied (the
1-step >=0.99 precision-budget STOP rule + the drill refuting all four streaming hypotheses).

### Results (real P150x4 + real pi05_base weights)
- R0 anchor (parent static-KV perstep): 0.991488 -> **0.999192** (improved, no regression).
- R0.5 oracle (streamed kernels, DRAM weights): velocity 0.991 -> **0.999178**, integrated 0.902
  -> **0.996886**.
- e2e streamed (concat-KV, capture=True trace): **1-step 0.996886, 5-step 0.983811** (>=0.95 bar
  cleared at both; 1-step also clears the 0.99 target).
- Functional trace parity: `stream_euler(capture=True)` RETURNS the replayed x_t; the replayed
  5-step result (0.983811) == the eager `capture=False` 5-step result (0.983811) == golden within
  PCC. => replay == eager == golden (SC3 met via the e2e path).

### no-tracy mandate + always-on real-time profiler (environmental)
This build ships an ALWAYS-ON real-time device profiler (no env toggle found). On REPEATED
multi-chip trace-REPLAY drains it emits "Skipping zone with end < start" on cross-chip
negative-delta zones and HANGS (the iter-2 hang). A standalone `replay()` second-drain test
reproduces it. Mitigation: validate trace parity via a SINGLE capture+replay per driver (the
e2e path), NOT a second standalone drain. NEVER run the streamed path under `python -m tracy`.
The standalone double-drain `test_pcc_pi05_pipeline_traceparity.py` re-drain assertion is
documented-deferred (environmental profiler limitation, not a port defect).

### Deferred device validation (framework #2; 4-chip box)
- n=6 `run_expert_chain` drop-in equivalence: code-complete + LOGIC-checked (signature parity,
  n!=6 raise, expert-only bare-chain) -- NOT device-run (needs >=6 chips).
- P150x8: code-complete + LOGIC-checked (`_resolve_topology` 8-way, carve loud-error on
  parent<8, `(3,3,3,3,2,2,1,1)` summing to 18) -- NOT device-run (needs >=8 chips).
- tracy device-time profiling: DEFERRED (always-on RT profiler hang on multi-chip replay).

### Fabric reset/retry harness (SC5)
`tests/pcc/_fabric_harness.py` (reset_board / open_parent_with_retry / close_parent) added;
the streamed PCC + trace tests rewired to it. `tt-smi -r` + 5s settle between distinct
multi-chip invocations cleared the recurring "active ethernet core timed out" / FABRIC_1D
stalls. Fabric hygiene: always pair FABRIC_1D-on-open with DISABLED-in-finally;
Pipeline.release_all() before close_mesh_device; one streamed device test per process.

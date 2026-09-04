<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Landmines

Every entry below was found by hitting it during one prefill bring-up on a Blackhole Galaxy. Each
cost at least a working session. They are grouped by how they fail, because **the silent ones are
the expensive ones** — a loud crash costs an hour, a silent wrong answer costs a phase.

## Silent wrongness — passes, and is wrong

| Trap | Why it is silent | What to do |
|---|---|---|
| `getattr(cfg, "rope_theta", DEFAULT)` on transformers 5.x | The attribute does not exist, so `getattr` returns **your default** — a RoPE wrong at every position, no exception anywhere. `rope_theta` moved into `rope_parameters`. | Read theta/scaling through dict helpers, in exactly one place, and assert non-`None`. |
| Inheriting a template's explicit `fp32_dest_acc_en=False` | Polarity is **per-op**. A matmul's default already enables fp32 accumulation, so the flag only ever hurts: measured 96x worse at bf8_b, 1168x at bf16 — and the degraded block still cleared a loose 0.99 gate. | Pass an explicit config with `fp32_dest_acc_en=True` everywhere, and A/B it in-suite. |
| Deriving the SDPA program grid from the device grid | Passes **every** single-card gate and only fails at SP>1, when the ring op asserts `ccl_core_grid_offset.x >= sdpa_grid.x`. On a (12,10) grid the offset is 11, so a derived 12 fails and a pinned 8 passes. | Pin the SDPA grid at 8x8; assert `sdpa_grid.x <= grid.x - 1` at construction so it fails at build time. |
| A negative control run at the wrong input scale | Swapping a decoder layer's two norm gains — a genuinely wrong model — scores **0.99993 on `randn` input** and **0.66830 on real embedding-scale input**. The residual stream dominates when the input scale is unrealistic, diluting the sublayer error to nothing. Drive residual-block controls at the scale the model actually sees. |
| `output_hidden_states=True` as an HF oracle | Its last element is the **post-final-norm** tensor, not the last layer's output. Comparing your pre-norm hidden state against it costs a debugging pass. Use a `register_forward_hook` on each decoder layer instead — its `forward` returns a bare tensor. |
| A PCC-based negative control on a layout bug | A **rotated** head-to-column mapping still scored **PCC 0.9989** on a position-labelled probe. The control passes while the mapping is wrong. | For layout/mapping bugs the discriminator must be **bit-equality**, not PCC. |
| A mutual-PCC gate ("path A == path B") with no stated depth | Applied to a 32-layer accumulated statistic it measures depth, not the op. | Name the depth the threshold applies at. |
| `eager_attention_forward` with `attention_mask=None` | Applies only the mask handed to it — so the reference is silently **non-causal**. | Set `_attn_implementation="eager"` **and** pass an explicit causal mask. |
| `from_pretrained` for a torch reference | Loads at the checkpoint's `torch_dtype` (bf16), so the reference shares the device's rounding and reports a flattered PCC. | Build modules bare and `.float()` them; keep the reference fp32. |
| `write_kv_chunk` handed a multi-head tensor | Writes only head 0. | Slice per head, one call each. |
| A gate that runs on a mesh the deployment never uses | It can be testing a configuration the model **cannot produce**. One KV-cache gate passed at TP=1 with a head count the model never emits at that mesh. | Run at least one gate at the deployment mesh before believing the cheap ones. |

## Loud, but the message points somewhere else

| Trap | Symptom | Cause |
|---|---|---|
| Two overlapping live submeshes | **Machine-wide hang that poisons the box** — every later collective hangs, including ones that just passed, until `tt-smi -r`. Produced false failures elsewhere and one wrong diagnosis. | Call `parent.quiesce_devices()` between submeshes. The worst landmine in the set. |
| A top-level partial mesh | Fabric-init timeout (`fabric_firmware_initializer.cpp:200`); not fixed by `RELAXED_INIT` or a torus descriptor. | `create_submesh` from the full mesh instead. |
| `TT_FATAL: cache and input num-heads dim must match` | Names neither TP nor the mesh. | A packed KV cache holding one KV head per chip forces **TP == num_key_value_heads**. |
| `RuntimeError: index N is out of bounds` from inside `gather_cos_sin` | Names neither RoPE nor chunking. | A contiguous-RoPE builder has a `start_pos <= seq_len` ceiling; chunked prefill needs the indexed builder. |
| `TypeError` on the **first served chunk** | After mesh open, weight load and `compile()` — the expensive way to find a signature mismatch. | The engine passes more kwargs than its own docs list. Assert against the real call site with an AST test. |
| `FrozenInstanceError` at runner startup | The engine **mutates** the config your adapter returns. | Return a plain subclass, not a frozen dataclass. |
| `ttnn.rms_norm_post_all_gather` raising `TypeError: incompatible function arguments` | The template you are told to copy passes its stats tensor **twice** — positionally (which is already the `stats` parameter) and again as `stats=`. `models/demos/gpt_oss_d_p/tt/rms_norm.py:82` and `:89`. That branch is dormant upstream, so nobody has run it; copy it verbatim and the failure lands the moment a TP-sharded residual is enabled, phases later. Pass `stats` once. |
| Ring collectives hang rather than error | Fabric mode defaulted to non-ring while every collective is `Topology.Ring`. | Pin the ring fabric mode; a manifest cannot set the mesh-descriptor path, so set both. |
| A probe "fails" at 257 tokens | `bfloat16` is exact only to **256**; 257 rounds to 256. The cache was correct, the probe was not. | Keep integer-valued probe payloads <= 256, or split across lanes. A failing probe is not evidence of a failing module until the probe's own numerics are checked. |

## Repo hooks that will block your commit

Found by a first independent run: these are enforced by `pre-commit`, so they fail at commit time
rather than while you write, and one of them is easy to hit hundreds of lines deep in a test file.

| Hook | What it rejects | What to write instead |
|---|---|---|
| `prefer-expect-error` (`.pre-commit-config.yaml:51`) | **any occurrence of the string `pytest.raises` in a `tests/` file** — it is a `pygrep`, so it fires on **comments and docstrings** too, including a docstring sentence explaining that the file uses `expect_error` *instead*. The documented same-line escape is wrong for prose (it claims an exemption for a line containing no call); reword the prose instead. | the repo-root `expect_error(ErrorClass, "substring")` fixture (`conftest.py:948`). The `message` argument is **mandatory** and must appear in the real error text, and the test must take `expect_error` as a parameter. |
| the `expect_error` fixture's `message` | Its docstring says the message "must appear in the real error text", implying a substring — but the implementation is `pytest.raises(error, match=message)`, i.e. a **regex**. `TT_FATAL` and assertion text is full of parentheses, so a literal message silently fails to match and the test reports `Regex pattern did not match` while the code is correct. Match on a metachar-free substring, or `re.escape` it. |
| `check-large-files` | files over 500 KB | **gzip** an oversized raw gate log rather than trimming it — compression is lossless, so the evidence stays byte-exact. On a 32-device machine the inflater is not progress bars: it is tt-metal's own `Pinned source memory start address ... must be aligned 64 B` warning, ~5,120 lines / ~800 KB **per gate log regardless of the gate**. Expect most multi-device gate logs to exceed the limit. |
| `black --line-length 120`, `isort`, `autoflake` | formatting, import order, unused imports | Run `pre-commit run --files <your files>` **before** recording any `path:line`, and **before starting a long device run** — reformatting mid-regression invalidates it against the tree (one session lost 22 minutes that way). Note `isort` may not be installed in `python_env`, so the hook is not reproducible by hand; run it through `pre-commit`. |
| `trailing-whitespace`, `end-of-file-fixer` | both, including inside raw logs | expect the first commit attempt to fail, be fixed by the hook, and need re-staging. |

## Method traps

| Trap | Why it bites |
|---|---|
| Copying a PCC threshold from another model's README | It is a guess. Every threshold set that way in this run was 1-2 orders of magnitude too loose. |
| Comparing PCCs across two different tests | Different inputs and different reference precision make them incomparable. This produced a confident, wrong conclusion that a full-layer PCC "launders" a sublayer — measured attenuation was only 1.1-1.7x. |
| Writing a guard from a parameter's **name** rather than its observed payload | Three such guards passed a static contract audit and were caught only by running the engine's real branches: a refusal on a parameter the engine always sends, a guard demanding a `dict` where the engine supplies a **list of dicts**, and two that compared a value with itself so protected nothing. Assert every guard can fire, and take every branch you claim to support at least once. |
| Assuming the noise floor models everything | It does not model a fused kernel's interior. SDPA alone sat 71x off its floor and accounted for the entire block gap, while hand-written stages sat at 1.0-1.5x. Attribute before blaming. |
| Committing or renaming while a phase session is live | Mislabels history, breaks in-flight path references, and rewrites raw-log provenance. |
| Verifying a rename with a smoke test | Import checks and a citation pass prove the tree is wired; they prove nothing about the gates. Re-run the previous phase's **gates**. |

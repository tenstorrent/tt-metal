# Plugin <-> model interface for structured outputs with speculative decoding (M1/M2)

Owners: plugin + CPU helpers = plugin agent (branch ssinghal/dsv4p1-structured-plugin, helpers in ssinghal/dsv4p1-structured);
model side (persistent mask buffer, masked split round, device tests) = model agent (branch ssinghal/dsv4p1-structured).
Copy of this file is committed at wt/structured/models/demos/blackhole/deepseek_v41_flash/STRUCTURED_SPEC_INTERFACE.md.

## What is already there (plugin agent, committed)
* `tt/grammar_mask.py` (torch only, CPU tested by `tests/test_grammar_mask_cpu.py`; run with
  `/mnt/tt-data/ssinghal/kit_root/python_env/bin/python -m pytest --noconftest <file>`):
  `unpack_mask`, `mask_bias`, `apply_mask_reference`, `check_rows_allow_something`, and
  `permute_to_runner(block, rows, runner_rows, n, kv=None, column0_only=True) -> int32 [runner_rows * n, 4040]`.
  The model calls `permute_to_runner` inside `_spec_verify` with `rows = [(x[0], x[3]) for x in rows]` (logical row, runner row; same pairs as `X[rr] = tokens[ii]`),
  `runner_rows = B` (the runner's B), `n = kv + 1`, `kv` = the adaptive verify k. Result row `r * n + j` = block row j of runner row r (user-major, the layout of `a_in`/ctrl).
  Idle / unmapped rows and columns >= 1 (M1) are all ones (-1). Empty-row masks raise ValueError (do not catch).

## Plugin -> model call
`decode_forward(tokens, start_pos, ..., spec_mode="argmax_ids", num_valid_drafts=..., slot_remap=..., sampling_params=..., grammar_bitmask=<tensor>)`
(arrives in `_spec_verify(**kwargs)`, today swallowed by `**kwargs`).
* `grammar_bitmask`: torch.int32 `[W, n_plugin, 4040]`, W = the plugin's padded batch width (same row index i as `tokens`), n_plugin = K+1 (the full block width,
  before the adaptive slice; the model slices to kv+1 via `permute_to_runner(kv=...)`). Bit b of word w = token 32 w + b allowed. All ones (-1) for rows without a grammar.
  M1: only column 0 is meaningful (structured rows carry no drafts: nv = 0, forced m = 0, commit = masked sample of column 0); columns >= 1 are all ones.
  M2: columns j = grammar state after j drafts (all ones past the request's own rows).
* Only present (non-None) on a verify step with >= 1 structured live row. Absent / None = the model must behave exactly as today (no mask, no extra work, single-trace round
  when no sampled rows).
* `sampling_params` as today (may be None when every row is greedy). With a mask present and `sampling_params is None`, treat all rows as greedy.
* Grammar is only known at sample time, so on such a step the plugin runs the WHOLE verify call from `sample_tokens` (not `execute_model`). Trace A could overlap the grammar fill; that is a later optimisation (the DEVICE_GRAMMAR_CONTRACT split), not part of M1.

## Model obligations
1. Capability: `model_capabilities["supports_verify_grammar_bitmask"] = True` (class and instance) AND env `DSV41_STRUCTURED_SPEC=1` (model switch, default 0 until validated).
   Instance attribute `verify_grammar_enabled: bool` set after load (False if the layout cannot serve it, e.g. the buffer could not be allocated): the plugin reads it once at runner init
   and keeps M0 behaviour (whole batch plain decode while a structured row lives) when it is False/absent.
2. Persistent buffer `int32 [rows * T, 4040]` (T = U*n per mesh row), `ShardTensor2dMesh dims=(0, 1)`, allocated BEFORE the first trace capture (all runners / buckets / adaptive k), updated in place
   with `copy_host_to_device_tensor`; reset to all ones (or skipped) on steps without a mask. See the design in the `tt/grammar_mask.py` docstring.
3. Apply the bias once in trace S before `samp.run` (shared top-32 sampler must see masked logits). A step with a mask always runs the split A/S/B round (greedy rows = greedy sampler params = exact argmax).
4. Return `VerifyOutput(argmax_ids=[W, n])` exactly as today; the committed token of a structured row is `a[r, 0]` (masked).
5. Do not touch prefix_cache.py / APC paths.

## Switch names
* plugin: `VLLM_TT_STRUCTURED_NARROW` (default 1; M0: structured on plain decode), `VLLM_TT_STRUCTURED_SPEC` (default 0 until validated; M1 per-row k=0, needs model capability + `verify_grammar_enabled`).
* model: `DSV41_STRUCTURED_SPEC` (default 0).
* Both on = M1 path; plugin on + model off = M0 path; plugin off = M0 path.

## Plugin-side behaviour of the M1 path (plugin agent)
`_request_is_argmax_certifiable` accepts a grammar-only structured request (no logprobs / allowed_token_ids / bad_words / min_tokens) when the M1 path is enabled; `_publish_draft` offers it no drafts;
the step still verifies for the other rows; `spec_grammar_bitmask_for_tt_batch(width=K+1)` builds the block; the verify forward is deferred to `sample_tokens`.

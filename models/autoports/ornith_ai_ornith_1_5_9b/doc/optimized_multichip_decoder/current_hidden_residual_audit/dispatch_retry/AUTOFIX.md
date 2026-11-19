# Hidden-residual fused gather: scalar-prefill dispatch repair

**Source/API repair prepared; no devices run.** New entry point:
`tests/multichip_current_hidden_residual_dispatch_retry.py`.
All original candidates, production, kernels, quality gates and job lists remain unchanged.

## Diagnosis

`../../current_hidden_fused_all_boundary_32.log` reports fused AG K=16384 versus weight K=4096. This is not evidence that the local width1024 packed decode family cannot run.

Production `tt/multichip_decoder.py:938`–`950` handles a non-aligned full-attention continuation one logical token at a time through `self.decode_forward`. The inherited scalar forward at line920 calls generic `norm`, whose sharded branch gathers normalized width1024 to width4096 (`:601`–`614`). Attention calls generic `linear(...,"packed_input")`; production generic linear at `:570`–`571` dynamically calls `self.decode_linear` for one logical row. The candidate's fused-gather branch then gathers width4096 again, leading to K=4096*4=16384 against the local packed-input weight's K4096.

The measured packed decoder follows a different path: it calls candidate `decode_norm`, which intentionally returns local width1024 when fused gather is selected, then candidate `decode_linear` directly. Its required K is1024*4=4096. B1 packed decode uses this same valid local-width contract and must keep fusion.

## Minimal width-aware fix

The new subclass overrides only `decode_linear`:

```python
if self.fused_gather_enabled and name == "packed_input" and x.shape[-1] == 4096:
    return MultichipDecoder.decode_linear(self, x, name)
return super().decode_linear(x, name)
```

The affected scalar prefill call now uses the ordinary DRAM-sharded projection on the already-gathered activation. Width1024 packed calls retain fused AG. The explicit `fused_gather_enabled` condition also leaves the passing separate/BF8-persistent path byte-for-byte in its original dispatch; in particular, it preserves that candidate's explicit BF16 projection output dtype instead of relying on the production call's default dtype.

A blanket bypass of all scalar row fusions is unnecessary. Production `reduce` at `:405`–`407` returns an already-scattered width1024 tensor unchanged for a sharded residual. The current fused-scatter implementation pads scalar input to32, uses its setup-owned `[1,1,32,1024]` output bank, then slices to logical1. Thus scalar attention output and MLP down projections can keep fused RS safely; generic reduce does not reduce them again. This matches the passing scatter-only short-run evidence reported by the parent.

Generic `linear`, `prefill_forward`, `decode_forward`, `norm`, `finish` and `reduce` remain inherited. Packed `decode_batch_packed`, decode norms/finish, current attention/GDN, persistent banks, current CCL grid, and residual layout are unchanged. Both measured layer boundaries remain local width1024; no inter-layer collective or replication restoration is added. Non-aligned public sequence lengths remain supported by the existing scalar continuation path.

## Host checks

`host_verify.py` exercises30 policy/batch/kind cases: gather, scatter, fused-all, persistent-all, BF8-persistent × B1/B6/B32 × linear/full attention. Tensor/device operations are mocked; actual method dispatch and persistent-buffer shape assertions run. Checks cover:

- Full-width scalar packed-input projection produces the expected local projection shape without AG.
- Scalar fused attention/down projection keeps padded32 persistent banks, slices to logical1, and inherited reduce returns the same tensor without another collective.
- Packed local-width normalization still invokes fused AG where selected, including B1.
- Generic prefill/finish/reduce method identities remain inherited; packed dispatcher is unchanged.
- CLI help is executed with device-opening entry points disabled.

`host_validation.json` records results. These are dispatch/shape checks, not numerical, JIT, Watcher or performance evidence. Parent should queue fresh labels using the same current-hidden-family arguments and references, replacing only the module with:

```text
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_current_hidden_residual_dispatch_retry
```

The wrapper patches the selected class only within its own process and restores the original binding in `finally`. A further API/kernel failure remains work under AutoFix; the material fused family is not rejected here.

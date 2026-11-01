# One-slot (Up=1) prefill context for short prompts (`DSV41_PF_UP1_SHORT=1`, default off)

**Why.** B=8 runs `DSV41_PREFILL_UP=2`: every chunk replay processes 2 x 512 row-tokens per mesh row, so a lone short prompt pays for an idle second slot. The window is 0.4-0.5 s on the device at 40 layers
(10-layer build, call timing dict: replay 150 ms with Up=2, 90 ms with the one-slot context). The existing S_pad "small bucket" is not used by the B=8/1M catalog default (single chunk trace, `DSV41_PF_SINGLE=1`; the
single trace already bounds its cost by the chunk position, not by S_pad): the fixed cost is the padded window itself.

**What.** A second prefill context (`PrefillModel.compile_dyn_up`): own carried tables at S_pad 16384 (`DSV41_PF_UP1_SPAD`), own input / hand-off / head buffers, own chunk trace(s) A/B, one prefill slot per mesh row.
`use_dyn(S)` swaps the whole bundle (`_UP_PM`, `_UP_SINK`, `.U` of the attention / sparse / indexer objects) with the context. Built in `warm_serving` before the first capture. Selection (`tt/pf_up1.py`, CPU tests
`tests/test_pf_up1_cpu.py`): every user of the call starts at 0, the call holds its whole prompt (end <= chunk, plugin `prefill_final`), at most one user per mesh row, no image prompt, no host-tier snapshot to restore, and with the
paged cache only prompts shorter than one block (it commits whole blocks and reads the carried tables by decode user). Anything else runs the Up=2 trace unchanged. The hand-off (pool pages, rings, compressor state, drafter taps,
Engram history) goes through the same sink by decode user; the key hand-off is recorded lazily with its context and flushed inside it before the first decode. No host-tier snapshot is taken for these prompts; the Up=2 slot
bookkeeping of the users is dropped. `DSV41_PF_UP1_FILE` (file holding 0 / 1) switches it live for A/B in one server.

**Results (B=8 / 1M, rc3 image, .47, wt/pf_small a3c63b2771f).**
| TTFT, client side | Up=2 (off) | Up=1 context (on) |
|---|---|---|
| ISL 17, c1 | 561 ms | 370 ms |
| ISL 128, c1 | 586 ms | 372 ms |
| ISL 512, c1 | 674 ms | 410 ms |
| ISL 1024 / 4096, c1 | 1162 / 3487 ms | 1179 / 3530 ms (not eligible, unchanged) |
| ISL 128, c4 / c8 (random placement) | 1258 / 1662 ms | 921 / 1423 ms (the first call of a wave is eligible) |
Adapter-level (harness, one user per mesh row, ISL 17 / 128 / 512): 598 / 598 / 719 ms -> 406 / 405 / 457 ms. Two users in a row, ISL >= 1024: identical.
* Boot DRAM: after the one-slot trace 406.8 MiB/bank free (410.8 before), end of build 400.2 MiB/bank (410.4 before): -10 MiB/bank; trace region +6 MiB/bank.
* Structured churn (8 plain users + 4 looping structured workers, M1): plain 27.2 -> 33.9 tok/s/user, structured request 10.2 -> 7.2 s. Plain decode alone 46.1 / 45.4 tok/s/user (c8, no change).
* Correctness: needle 1k / 8k 4/4 each; GSM8K 32 slice 26 / 26 (off, twice) vs 27 / 28 (on) correct; greedy completions of repeated runs differ in the server regardless (off1 vs off2 13/32 identical, on vs off 12-18/32),
  so identity is not a usable gate; first-token logits (model level, 40 layers, random prompts) differ by up to 1.8 between the contexts, the same size as the slot-to-slot difference inside the Up=2 trace (same prompt in slot 0 / 1:
  0.5-2.3 at 10 layers); cold-vs-cold in one context is bit-identical. Hang canary 15 x (8 x 8k + structured rows) clean.
* Not done: paged-cache (`DSV41_APC_PAGED=1`) device run; C=128 window (lever 2).

**rc4b validation (2026-10-31/11-01, .34, image rc4b-0467f96ebb2).** Found and fixed: `PrefillModel._vis_dirty` (the "clear the vision buffers after an image window" flag) was keyed by the chunk size and reset at prep time, so a call on the one-slot context cleared the flag while the Up=U context kept a stale image mask / embedding override; the next text-only window of the Up=U context (any prompt >= 512 tokens, or two users in a row) saw image rows (text chats answered with `<ref>..</ref><box>..` grounding text; 1-2% of multi-turn chats in a mixed image + text run, 0 with the lever off). Deterministic repro (image request, short text request, 1200-token needle prompt): needle lost 24/24 before, 0/24 after; the flag is now per vision buffer and decided at upload (`upload_inputs`). Measurements and the decision are in the rc4b_pfsmall report.

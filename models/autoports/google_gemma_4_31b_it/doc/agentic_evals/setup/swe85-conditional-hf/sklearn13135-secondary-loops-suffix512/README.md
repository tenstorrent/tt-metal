# Sklearn13135 secondary-loop HF controls (candidate only)

This package proposes two serial pinned-HF reference controls for the exact saved 512-token inputs used by full-mode call29 seeds42 and43. Each has chosen64 teacher-forcing positions and a512-token free-generation cap. The manifests reference the existing full-mode fixture IDs in place; prompts are not rendered or altered.

The full-mode targets are secondary loop families: seed42 repeats an unchanged “Since I’ve fixed the source code / check” sentence; seed43 enters an apostrophe-a token loop after a brief fix phrase. They differ from the original 17,395-token missing-tool behavior and do not establish native-prompt equivalence. Full seed44 was normal at85 tokens. The served comparison now uses the root-reviewed correction at `sklearn13135-suffix512-head-bfp8-correction/runs` with selected `stage12_head_bfp8_lofi`: seeds42/44 reproduce the native apostrophe-ve family and seed43 is an apostrophe-hyphen loop. The previous served-path proposal/commands are preserved with stale labels.

`run_secondary_hf.py` is inert unless root admits the exact candidate and invokes one explicit `--group N`. Group2 requires a root review binding group1 launch, completion, raw output hashes, and forecast result. There is no automatic progression. Host preflight requires zero device owners, no HF runner process, and at least100GiB available memory. Saved generation commands are not used for validation.

The host-only forecast uses eight prior pinned-HF 512-token case timings: 1220.183 seconds per process [980,1780], including a180-second load margin; each has a3600-second hard timeout. These are planning estimates, not serving measurements. No command or HF inference has been run from this package.

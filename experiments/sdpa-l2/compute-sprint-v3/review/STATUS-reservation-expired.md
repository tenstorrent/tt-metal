# Reservation-expiry handoff

September 18, 2026, after root confirmed no active IRD reservation at23:56 UTC.

- Last launched B job: `qualify_valid.py --label B-valid-k3-01 --q-length 1024
  --k-length 1536 --cores 2 --distributions
  normal,uniform,constant_v_scaled,common_v,identity_transitions,repeated_kv,zero_v`.
- Existing local unified-exec session31103 still reported running with no new
  output when polled. No fresh SSH command was launched after the pause.
- Remote completion is unknown. No B-valid JSON or raw log was downloaded.
  No B-valid timer was launched.
- All prior15 B JSON files and matching raw logs are local. Independent local
  audit passed198 variant records, all numerical/replay gates, and every
  recorded source hash against current files.
- Original, direct, replay and validity adapters are frozen. Shared group2
  implementation ownership remains with the compensated-kernel agent.
- No local/remote process termination, reset, lock mutation or recovery action
  was taken. Root coordinates reservation reacquisition and any stale-session
  cleanup.

Frozen B-valid adapter SHA256:

| File | SHA256 |
|---|---|
| `qualify_valid.py` | `7c8aea86dc7acf0b0ec0939ab205b5d88daaf8e2e9c72546b7eaad8fdd10dbb6` |
| `bench_valid.py` | `286843dd395dfb65696c810456ffc4af78b3e37389accdb58aa4ba414bf1eec5` |
| `compute_valid.cpp` | `35a14ca7ad84005e28c76ca4b116ef046d0be2bfeafa36770a8eb6a0ba0c5696` |
| `resident_valid.cpp` | `9b114aa9d8adc72f468876193c3ca213d6fc014eead2de3db9bcc47076903ee0` |

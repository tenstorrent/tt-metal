# Archived data — Wormhole, non speed-of-light

Raw `noise_report.points.csv` files are ~21 MB each and do not compress, so only
the parts a follow-up needs are kept here. Regenerate the rest with the commands
in `../../../README.md` section 11.

| file | what it is |
|---|---|
| `flagged_l1_to_l1_x10.csv` | Every point the `>2% AND >30 cycles` rule flags in the ten-run L1_TO_L1 baseline, with all ten run values and the full sweep configuration |
| `top_movers_l1_to_l1_x10.csv` | The 1,000 largest movers, for anyone examining the tail below the rule |
| `summary_by_test_l1_to_l1_x10.csv` | Per (test, marker) counts above 0.5 / 1 / 2 percent and the worst move — the population statistics without the bulk |
| `serial_replay_l1_to_l1.csv` | Ten replays of one recorded order at `-n 1`, merged per point. The evidence for section 8.8 |

Provenance is in `../noise_l1_to_l1_x10_metadata.txt` and `../machine_x10.txt`.

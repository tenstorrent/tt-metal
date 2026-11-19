# PLE scale-byte quantization risk check

The group20 BF16-scale representation fails the unchanged 0.995 per-row GDN
gate at the advertised 262,144-token context. The chosen group10 representation
uses four-bit weight codes and eight-bit scale IDs at the same embedding-table
payload size. It passes the full continuous-state CPU replay and the existing
short full-decoder holdouts. These CPU checks retain exact BF16 experts and
exclude MoE from the long-context state path. Separately, the complete resident
TTNN layer1 decoder has passed the short device contract with BFP8 experts;
that result does not yet establish advertised-context full-decoder acceptance.

All comparisons use the original exact BF16 checkpoint reference. The long
fixture is `/home/ttuser/qwen38-flash/ple_cpu_prefill_262144.pt`, prepared from
snapshot `812d96e10a0dd82b69d749e68693698cf6c36daa`, seed12031, hidden standard
deviation0.02, and HF chunks of1024. The replay starts with fresh state, carries
all PLE/GDN state through every prefix chunk, and finishes with the single
decode token at position262143. An exact-embedding control reproduces every
saved output boundary of the first1024 tokens bit-for-bit. Hash IDs are checked
exactly for every replayed token. No TTNN import or device run occurs.

| Representation | Minimum PLE PCC | Minimum GDN attention PCC | Minimum attention contribution PCC | Gate |
|---|---:|---:|---:|---|
| Group20, BF16 scales | 0.996726573 | 0.994461000 | 0.994281590 | Fail:3 GDN rows |
| Group10,256 positive log scales | 0.997705579 | 0.996126771 | 0.996032298 | Pass |
| Group10,zero +255 narrow log scales | 0.997689009 | 0.995980263 | 0.995969534 | Pass |
| **Group10,zero +255 wide log scales** | **0.997686565** | **0.995966911** | **0.995944798** | **Pass** |

Attention contribution means `after_attention - after_ple`, subtracting each
path's own PLE residual in FLOAT32. The group20 worst GDN row is199012; first
and final4096-token windows both pass and would miss the failure. The chosen
wide candidate's worst gated row is356. Its mixed attention input, measured
diagnostically, has minimum PCC0.995583653 at245032. No failing positions were
removed from any sweep. Evidence is in `ple_context_group20_cpu.{json,log}`,
`ple_context_group10_scale8_cpu.{json,log}`,
`ple_context_group10_scale8_zero_cpu.{json,log}`, and
`ple_context_group10_scale8_wide_cpu.{json,log}`.

The S65/seed1753 full-decoder holdout passes prefill, three changed decode
tokens including EOS, fresh lengths1/31/32/33, and eight-token continuations
starting at1/31/32/33. Its worst residual-subtracted contribution PCC is
0.996312916. The original S33/seed1731 fixture passes with prefill contribution
0.996155739 and decode contributions0.997771621/0.998090327/0.997727334.
See `ple_decoder_group10_scale8_wide_holdout.{json,log}` and
`ple_decoder_group10_scale8_wide_original.{json,log}`.

The subsequent full runtime contract in `ple_group10_contract.json` passes with
the complete resident table, real nonembedding weights, and explicit BFP8
experts. Lengths1/31/32/33/65/1 have minimum whole-output PCC0.996588647 and
contribution PCC0.996302247. Eight-token continuations from1/31/32/33 have
minimum contribution PCC0.996451795, and every dropped-history negative control
fails the same gate. Three changed traced decode inputs have contribution
PCC0.997512937/0.998091578/0.997905254. Output and all four state buffers are
bitwise repeatable after reset. Prepared prefills also run with the decode
trace live. The strict runtime audit passes39 calls, trace allocation tracking
is enabled, and the run exits0 with watcher evidence. Artifacts are
`provenance/ple_group10_contract.json`, `logs/ple_group10_contract.log`, and
`watcher/ple_group10_contract/`; the provenance records the exact command,
environment, and source hashes. This is short-context device acceptance;
largest feasible public prefill and full-layer advertised-context decode remain
separate gates. Host wall timings in this contract are not device profiler
performance evidence.

The immutable weight codebook and scale LUT are calibrated from8192 checkpoint
rows,64 per shard across all128 shards, using seed43191. All calibration IDs
are disjoint from every hash ID in the long fixture and both short holdouts.
Only checkpoint values influence calibration; no activation or failed token
selects the codebook. The chosen LUT reserves ID0 for exact zero and spans
positive scales0.0003147125244140625 through0.1689453125. Those limits expand
the independent training extrema by factors8 below and4 above before BF16
rounding. Uniform and logarithmic scale encodings are compared on the same
independent calibration rows in `ple_group10_scale8_wide_hypothesis.json`.
All training IDs, fixed levels, and calibration provenance are retained there.

Each160-value row contains20 UINT32 weight words and4 UINT32 scale words.
Weight words pack eight consecutive4-bit codes, low nibble first; scale words
pack four consecutive8-bit IDs, low byte first. Both streams use the same
32-stripe layout: linear element `i` is stored at `[i %32, i //32]`. A scale
word address is `row*4 + lane`, at most1,280,006,143 for the full domain.
Each decoded scale serves10 consecutive weight values. The product of its
BF16-exact scale and FLOAT32 weight-codebook level is rounded to BF16.

| Full-domain payload | Bytes |
|---|---:|
| 320,001,536 rows ×80B weight codes | 25,600,122,880 |
| 320,001,536 rows ×16B scale IDs | 5,120,024,576 |
| Combined table streams | 30,720,147,456 |
| Logical fixed codebooks, excluding device layout replication | 576 |

The format is `u4_grouped_u8_scale_lut_stripe32_v3`, with
`scale_encoding=uint8_fixed_log_lut`, `group_size=10`, `scales_per_row=16`, and
`scale_words_per_row=4`. The immutable FLOAT32-byte hashes are:

- Weight codebook: `59c463a9a75bec1e953be5a908eec43d0d0b8ff1cb68d3fd8238bbebaf41e68f`.
- BF16-exact scale LUT: `180b4e3289539394e3ece69d1268ec7e30a7c920c3055bfad89fa70253b5dbf6`.

CPU packing matches independent nearest-level reconstruction exactly on1152
real rows from all128 shards, including first/last and stripe/chunk boundaries,
plus64 synthetic rows including zero. Serialization and stripe retrieval are
exact, and the maximum scale address is checked. Legacy reconstruction remains
exact across17 previously exercised settings. See
`packed_embedding_scale8_cpu_parity.{json,log}` and
`packed_embedding_scale8_legacy_regression.{json,log}`.

The complete cache is prepared and verified separately at
`/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide`; group20 and
legacy absmax caches are preserved. The preparer uses8 CPU threads and chunks
of65536 rows, and records fitted-scale extrema and counts below/above the LUT
range over all5,120,024,576 groups. The1440 below-range groups are exactly the90
padding rows beyond the valid hash domain of320,001,446 rows; their maximum
absolute checkpoint value is1.9651164376299768e-18. There are **zero below-range
or above-range groups in the valid hash domain**. The full-table maximum fitted
scale is0.08966350555419922, below the LUT maximum0.1689453125.

Read-only verification of1152 boundary rows across all128 shards matches both
complete-cache streams exactly. An altered LUT is rejected by cache identity
validation. The full stream checksums are recorded in
`packed_embedding_group10_scale8_cache_verification.{json,log}`; preparation
evidence is `packed_embedding_group10_scale8_preparation.{json,log}`. Checksums:

- Manifest: `642f0e2382b182aba6d1eee24fc2e1ab1ef53d84d30b7284a246cbb7d1680792`.
- Words: `4a0b66d12837e63437f1e02c0bdd16a2d6a2317d07eda561a9308d5bbc9b7b3b`.
- Scales: `2d34f192bb5e7e6f12fb37550c980032e4e5311806da8d21c4001216fa5e09e3`.

A manifest appears only after all128 shards are written and flushed. This cache
completion is now accompanied by the short full PLE device/trace contract
above. Capacity and advertised-context tests must still validate the final
runtime source. The existing long CPU fixture has a bitwise-equal formatted
replacement, `ple_cpu_prefill_262144_formatted.pt`, for current harness source
identity; see `ple_cpu_fixture_format_parity.json`.

Reproduction commands (run from the checkout, CPU only):

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=8
PLE_TESTS=models/autoports/qwen_qwen3_8_flash_next/tests
PLE_DOC=models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder
PLE_SNAPSHOT=/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa
PLE_FIXTURE=/home/ttuser/qwen38-flash/ple_cpu_prefill_262144.pt

python_env/bin/python "$PLE_TESTS/probe_ple_scale8_hypothesis.py" --fixture "$PLE_FIXTURE" --reserve-zero --lower-expansion 8 --upper-expansion 4 --output "$PLE_DOC/ple_group10_scale8_wide_hypothesis.json"
python_env/bin/python "$PLE_TESTS/probe_ple_context_quantization.py" --fixture "$PLE_FIXTURE" --spec "$PLE_DOC/ple_group10_scale8_wide_hypothesis.json" --output "$PLE_DOC/ple_context_group10_scale8_wide_cpu.json"
python_env/bin/python "$PLE_TESTS/probe_ple_decoder_quantization.py" --snapshot "$PLE_SNAPSHOT" --scale8-spec "$PLE_DOC/ple_group10_scale8_wide_hypothesis.json" --methods scale8:10 --seq-len 65 --seed 1753 --lengths 1,31,32,33 --continuation-starts 1,31,32,33 --output "$PLE_DOC/ple_decoder_group10_scale8_wide_holdout.json"
python_env/bin/python "$PLE_TESTS/probe_packed_embedding_scale8_cpu.py" --snapshot "$PLE_SNAPSHOT" --spec "$PLE_DOC/ple_group10_scale8_wide_hypothesis.json" --prepare-dir /home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide --output "$PLE_DOC/packed_embedding_group10_scale8_preparation.json"
python_env/bin/python "$PLE_TESTS/probe_packed_embedding_scale8_cpu.py" --snapshot "$PLE_SNAPSHOT" --spec "$PLE_DOC/ple_group10_scale8_wide_hypothesis.json" --verify-cache-dir /home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide --output "$PLE_DOC/packed_embedding_group10_scale8_cache_verification.json"
```

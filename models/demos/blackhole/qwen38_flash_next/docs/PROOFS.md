# Release proofs

The runs that verified this release end to end on hardware, from a fresh clone, following the README as written.
Where a proof quotes acceptance divergence indices measured before the GDN gate fix of 2026-09-06, the current
indices are in `NUMERICS.md`.

## The QuietBox, 2026-09-04 (pinned build)

A QuietBox (`tt-quietbox`: 4x p150c, fw 19.4.1.0) served the model on 2026-09-04 from a pinned build: startup
acceptance 96/96 against the CPU, 19.6 tokens/s at 32k.  The launcher and profile are the ones in this repository.

## 4x p150, 2026-09-05 (expert conversion rate)

On a 4x p150 host the first start's BF4 expert conversion took about 33 s per layer (2.2 GB written per layer;
measured 2026-09-05: 25 layers in 812 s).

## The QuietBox, 2026-09-06 (fresh clone of the public repository)

The QuietBox (`tt-quietbox`: 4x p150c, which `tt-smi` reports as p150b; firmware bundle 19.4.1.0, tt-kmd 2.6.0-rc1, 32 cores, 503 GB RAM, Ubuntu
22.04, clang-20, Python 3.10.19 through `uv`) served the model on 2026-09-06 from a fresh clone of the public repository
at `cadebdff7c1c`, following the README's sections 2-5 as written (the deviations found on the way are folded into the
README):

- clone 53 s, the three submodules 15 s, `build_metal.sh` 684 s with its defaults, `create_venv.sh` 95 s; the
  runtime identity of every run: head `cadebdff7c1c`, tree `dd25966f522c`, clean, extension
  `f3d1fb4c3ab4...`.
- the checkpoint copy already on the host: `download_checkpoint.py --verify-only` verified 142 of the 145 listed files
  in 13 s (LICENSE differed, `.gitattributes` and `configuration.json` were absent), the plain run fetched those three in
  17 s (145/145); `verify_checkpoint_files.py`: 131/131 shards, 360,000,192,888 bytes, every SHA-256 equal to the
  ModelScope listing (176 s, 4 workers).
- `prewarm_ple_table.py`: the 104,298,732,704 B of the n-gram table in 13.4 s (already resident), 33/33 files resident.
- the first start (`--profile tt-quietbox --acceptance --require-json-96`, 32k): mesh open 9.6 s; the 49 BF4 layers
  1772.5 s (35.9-36.9 s each, 100 GB written); target build 41.4 s; warm pass with a cold JIT cache 150.2 s; captures
  5.1 s and the chunk capture 1.7 s; acceptance replay 59.2 s; `READY` 2096.6 s after the mesh open (launched
  17:19:30Z, `READY` 17:54:33Z); 474,261,568 bytes free per bank after the captures.
- acceptance: `json` 96/96 (the gate passed); the other eleven records leave the CPU stream at the same indices as the
  4x p150 hosts did (chat 8, code 24, fact 15, list 46, math 61, multilingual 9, prose 13, refactor 22, sky 19, story
  6, summary 75: the table before the GDN gate fix of 2026-09-06, `NUMERICS.md` has the current one); 19.4-19.6
  tokens/s in the replays.
- requests over the LAN: a 36-token answer at 19.1 tokens/s (first token 0.30 s after a 33-token prompt), a 128-token
  generation at 19.6 tokens/s (first token 0.39 s, 47-token prompt in 2 chunks); the CLI's question answered.  SIGTERM
  stopped it cleanly (`result.json` status `stopped`, mesh closed, launcher exit 0).
- the same launcher line with `--mtp 4` (warm caches; the MTP kernels compiled on this start): `READY` 225 s after the
  launch (MTP warm pass 40 s, acceptance replay 48 s); `json` 96/96 through MTP at 55.0 tokens/s (4.8 tokens per
  pass), the split hand-off gate passed in both orders; `code` left the CPU stream at 44 and `fact` at 16, the other
  nine records at the plain-decode indices of that day (before the GDN gate fix of 2026-09-06; `NUMERICS.md` has the
  current MTP indices); 375,594,496 bytes free per bank (98.7 MB less than without MTP).
  Requests: the `json` prompt as a chat request reproduced the CPU record's 96 tokens at 55.2 tokens/s; a 6942-token
  prompt prefilled in 23.4 s (3.37 ms per prompt token, 217 chunks of 32 rows) then decoded at 31.3 tokens/s (3.0 per
  pass), its follow-up turn reused the 6980 committed tokens (first token 1.6 s, 43.1 tokens/s); a 128-token generation
  36.7 tokens/s (27.0 ms per token); a 220-token prose answer 31.0 tokens/s (2.6 per pass).
- the same line with `--long-chunks` (warm caches; the 128-row chunk kernels compiled on this start): `READY` 214 s
  after the launch; acceptance identical to the plain start (`json` 96/96, the same eleven divergence indices, 19.4-19.6
  tokens/s); 439,384,896 bytes free per bank (34.9 MB less than without).  The 6942-token prompt prefilled in 18.8 s =
  2.71 ms per prompt token (first token 18.9 s; 3.37 with 32-row chunks on the MTP start above), hand-off 805 ms, the
  same answer; decode 50.6 ms per token (19.6 tokens/s on a 128-token generation).  The `json` chat request again
  reproduced the CPU record's 96 tokens.
- `--allocated-context 65536` (the 64k component and model I/O caches built on this start, 9.8 GB): `READY` 220 s after
  the launch (target build 41 s, warm pass 13.5 s, acceptance replay 60 s); `json` 96/96, the same divergence indices;
  `/health` `context_limit` 65472; 419,440,704 bytes free per bank.  Left serving the LAN on port 8000
  (`--serve-seconds 86400`); a client on the LAN got its first token 0.30 s after a 33-token prompt.
- Every start above was stopped with SIGTERM between runs and closed its mesh (`result.json` status `stopped`,
  launcher exit 0); no board needed a reset.

## A QuietBox 2, 2026-09-07 (mixed DRAM harvesting; a contributor's box)

A QuietBox 2 (2x p300c, four dies) served the model on 2026-09-07; the run, the measurements and the fix are
msollanych-tt's (pull request 7 of the public repository).  One die serves its DRAM banks from worker column 5, the
other three from column 6, and the release's BF4 guard, which required identical bank-to-worker coordinates on all four
dies, refused the conversion before it began.  The guard now compares what the packing consumes, the bank count and the
bank order per die (`SERVER.md`, the expert cache); with it the box converted the 49 layers (about 14 minutes, 100 GB),
reached `READY`, and its startup acceptance was `json` 96/96 with the eleven divergence indices equal to the pinned
table of the head it cloned (cadebdff, before the GDN gate fix: chat 8, code 24, fact 15, list 46, math 61,
multilingual 9, prose 13, refactor 22, sky 19, story 6, summary 75), identical on a second run after a board reset:
the differently harvested die reproduced the 4x p150 device streams.  19.9 tokens/s at 32k; 17.4 tokens/s end to end
with 256k allocated context.  The `qb2` route derived as `[0, 1, 2, 3]`.

## Our QuietBox 2, 2026-09-19 (the public tree; the fabric-order route)

Our own QuietBox 2 (2x p300c, four dies whose harvested Tensix columns differ die by die) served the model on 2026-09-19
from the public tree at 864c13307b, together with the route derivation that landed the same day (`archive/q38-qb2-route`,
66a67f0dc8): the fabric embeds the 1x4 line onto this ring as chips `(1, 0, 3, 2)` while the ring walk from chip 0 gives
`(0, 1, 2, 3)`, and with the walk's order the first all-gather found no forwarding direction; the profile now derives the
route from the fabric's chip order at start and `READY` records it with the walk beside it (`route_ring_walk_agrees`
false on this box).  With warm caches `READY` came 194 s after launch (prewarm 19 s, mesh and fabric 2 s, target build
21 s, cold JIT about 120 s, captures and the acceptance replay about 45 s): `json` 96/96, the twelve divergence indices
identical to the pinned table of the 4x p150 boxes (chat 43, code 32, fact 15, json none, list 56, math 61,
multilingual 9, prose 13, refactor 24, sky 19, story 6, summary 75; bitwise with them), 14 fused kernels,
465,231,616 bytes free per bank, 24.3 to 24.7 tokens/s at 32k in the replay (median 24.6) at that head.  The expert
cache converted the 49 layers in 15.5 minutes (18.8 s per layer) with the bank order `[4, 0, 2, 6, 7, 3, 5, 1]` on all
four dies.  One DRAM reader per bank: one die serves its banks from another worker column than the other three, and two
readers per bank need one bank-to-worker assignment on every die; the rule that decides this at build time (two readers
only when every die's assignment agrees, otherwise one, the reason recorded in `READY`) landed on 2026-09-22.  The host
ran with its IOMMU in identity mode; in the translated mode the first weight upload had stalled in the driver's page pinning.

## Our QuietBox 2, 2026-09-23 (the compact expert weight layout, four dies)

The same box served the model on 2026-09-23 with tt-metal's compact expert weight layout for `moe_compute` (the
owned-column layout, upstream pull request 57564, at that head): the acceptance table equal to the run above, `READY`
free bytes per bank 465,223,424 -> 1,611,841,280 (9.17 GB more per device: the expert weights 26.2 -> 17.0 GB per
device), served time per output token 40.03 -> 39.44 ms, replay decode 25.70 -> 26.12 tokens/s, prefill 3 to 6 %
faster, at that head.  A drained decode-step census on the same box the same day (20 traced steps, chip 0): span
39.02 -> 38.34 ms, kernel 34.61 -> 33.88 ms, `moe_compute` 3.54 -> 3.19 ms per step, 3,370 programs in both, every
other module equal.

## What the other profiles have

- `p150-line` (4x p150 in one host, ethernet line): the README's numbers were measured on 4x p150 hosts (the
  performance table of 2026-09-04, the pinned divergence tables of 2026-09-06 in `NUMERICS.md`, the conversion rate of
  2026-09-05 above); a fresh-clone run of the form recorded above for the QuietBox is not recorded for a p150 line.
- `qb2` (QuietBox 2, 2x p300c): the contributor's run of 2026-09-07 and our runs of 2026-09-19 and 2026-09-23 above;
  the route is derived from the fabric's chip order at start (`(1, 0, 3, 2)` on our box, where the ring walk gives
  `(0, 1, 2, 3)`) and recorded in `READY` with the walk beside it.

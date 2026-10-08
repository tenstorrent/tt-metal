# t277 — make DIFFVAE_NA_EDGE_ORDER the default, land on t48

- Land branch (local): ttp/t277-land = origin/ttp/t48-ltx25-integrated 747bc0612f1
  + 6aca3975c8d (cherry-pick of t274 ac876509daa; ablate_* enum/fn/hash entries dropped, t48 lacks them)
  + a5a774ea17f (flip: edge_order_requested() on unless =0; still in compute_program_hash;
    new unit test test_edge_order_is_bit_identical, not run: needs a 1x1 mesh open on a galaxy).
- blx01: driver /var/tmp/fasth3/t277/drv (go277.sh from t277-drv/), builds /var/tmp/fasth3/t263/b
  @a5a774ea17f (log /var/tmp/fasth3/t277/build_a5a774ea17f.log), then ONE broker job, arms
  default / off:DIFFVAE_NA_EDGE_ORDER=0, one process each, profiled, -t 240 (job 042 took 156 s).
  Out: /var/tmp/fasth3/t277/out (run.log, stage_tree_{default,off}.txt). Probe:
  `ssh g15blx01 bash /var/tmp/fasth3/t277/drv/probe.sh`.
- Expect: default ~2.48 s, NA ~94 ms/block; off ~2.71 s; md5 2797bc15ab69a49d945b26d85ff36675 both.
- Then: on ttp/t277-land `ttp push --detach` (rebase onto latest t48; #276 touches the same reader).
- 12:07 UTC: build ok (27 s incremental, transformer unity rebuilt). Broker job 043 submitted (-t 240).
  First launch failed before submit (script name typo, no device work).
- Job 043 done, no drops: default 2.313 s, off (DIFFVAE_NA_EDGE_ORDER=0) 2.531 s, one process each,
  both md5 be633a6944d3767c9d4818909ea23034 (seed 0).
- md5 judgment: be633a... is the t48 747bc0612f1 baseline. #273 job 021 seed 0 gave be633a... with
  DET_A2A on (t48 default) and 2797bc... with DET_A2A=0. #274 ran on 5833f56096f (no A2A), hence
  its 2797bc.... So edge order is bit-identical on t48: OK to land.
- Next: on ttp/t277-land `ttp push --detach`; then back on this branch `ttp push --own --detach`.

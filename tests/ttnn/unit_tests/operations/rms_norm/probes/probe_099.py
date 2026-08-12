from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main

main(
    variants=["prep_first", "after_push", "writer_prep", "cheap_after_push"],
    cores=110,
    c=3,
    mask=True,
    tag="i4_c3_tail_rep2",
    keep_log=False,
)
main(
    variants=["prep_first", "after_push", "writer_prep", "cheap_after_push"],
    cores=110,
    c=2,
    tag="i4_focus_c2_rep3",
    keep_log=False,
)

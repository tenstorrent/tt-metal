from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main

main(
    variants=["prep_first", "after_push", "cheap_after_push", "writer_prep"],
    cores=110,
    c=3,
    mask=True,
    tag="i4_c3_tail",
)
main(
    variants=["prep_first", "after_issue", "after_push", "cheap_first", "cheap_after_push", "writer_prep"],
    cores=110,
    c=2,
    tag="i4_focus_c2_rep2",
)

from ttnn.operations.rms_norm.perf_experiments.scaler_offpath.driver import main

main(
    variants=[
        "prep_first",
        "after_issue",
        "after_push",
        "cheap_first",
        "cheap_after_push",
        "writer_prep",
        "cheap_poisoned",
    ],
    cores=8,
    c=2,
    tag="i4_n8",
)
main(variants=["prep_first", "after_push", "cheap_first", "writer_prep"], cores=110, c=1, tag="i4_c1")

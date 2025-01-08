import ttnn


test_data = [
    {
        "input": (8, 8, 512),
        "kernel": (256, 3, 3),
        "stride": (2, 2),
        "pad": (2, 2),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.WidthSharded,
        "out_layout": ttnn.ROW_MAJOR_LAYOUT,
    },
    {
        "input": (8, 8, 512),
        "kernel": (256, 3, 3),
        "stride": (2, 2),
        "pad": (2, 2),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.WidthSharded,
        "out_layout": ttnn.TILE_LAYOUT,
    },
]

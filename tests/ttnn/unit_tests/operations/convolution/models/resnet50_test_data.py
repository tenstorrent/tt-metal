import ttnn


base_test_data = [
    {
        "input": (224, 224, 4),
        "kernel": (64, 7, 7),
        "stride": (2, 2),
        "pad": (3, 3),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (56, 56, 64),
        "kernel": (64, 1, 1),
        "stride": (1, 1),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (56, 56, 64),
        "kernel": (64, 3, 3),
        "stride": (1, 1),
        "pad": (1, 1),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (56, 56, 64),
        "kernel": (256, 1, 1),
        "stride": (1, 1),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (56, 56, 256),
        "kernel": (64, 1, 1),
        "stride": (1, 1),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (56, 56, 256),
        "kernel": (128, 1, 1),
        "stride": (2, 2),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.HeightSharded,
    },
    {
        "input": (28, 28, 128),
        "kernel": (128, 3, 3),
        "stride": (1, 1),
        "pad": (1, 1),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.BlockSharded,
    },
    {
        "input": (28, 28, 128),
        "kernel": (512, 1, 1),
        "stride": (1, 1),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.BlockSharded,
    },
    {
        "input": (28, 28, 256),
        "kernel": (512, 1, 1),
        "stride": (1, 1),
        "pad": (0, 0),
        "dilation": (1, 1),
        "bias": True,
        "shard": ttnn.BlockSharded,
    },
]

out_layout_data = [
    {"out_layout": ttnn.ROW_MAJOR_LAYOUT},
    {"out_layout": ttnn.TILE_LAYOUT},
]

test_data = [{**x, **y} for x in base_test_data for y in out_layout_data]

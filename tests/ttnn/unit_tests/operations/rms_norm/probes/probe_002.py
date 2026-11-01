run((1, 1, 6400, 64), layout=ttnn.TILE_LAYOUT)
run((1, 1, 6400, 1280), layout=ttnn.TILE_LAYOUT)
ttnn.close_device(device)

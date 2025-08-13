import ttnn


class OFT:
    def __init__(self, channels, cell_size, grid_height, scale=1):
        super().__init__()

        y_corners = ttnn.arange(0, grid_height, cell_size) - grid_height / 2.0

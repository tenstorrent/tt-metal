import ttnn

from ...layers.module import Module, Parameter


class Conv1dSimpleTTNN(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        mesh_device,
        dtype=ttnn.bfloat16,
        bias=True,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.weight = Parameter(
            total_shape=[
                out_channels,
                in_channels,
                1,
                kernel_size,
            ],
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.bias = Parameter(
            total_shape=[
                1,
                1,
                1,
                out_channels,
            ],
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self._weight_prepared = None
        self._bias_prepared = None

    def _prepare_torch_state(self, state):
        if "weight" in state:
            w = state["weight"]

            # torch Conv1d:
            # [out,in,k]
            #
            # TTNN:
            # [out,in,1,k]

            state["weight"] = w.unsqueeze(2)

        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, 1, 1, -1)

    def forward(self, x):
        # x:
        # B,T,C

        B, T, C = x.shape

        x = ttnn.reshape(x, (B, 1, T, C))

        weight = self.weight.data
        bias = self.bias.data

        # conv1d exige pesos host em ROW_MAJOR durante preparação
        if weight.layout != ttnn.ROW_MAJOR_LAYOUT:
            weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)

        if bias.layout != ttnn.ROW_MAJOR_LAYOUT:
            bias = ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT)

        if self._weight_prepared is None:
            y, _, prepared = ttnn.conv1d(
                input_tensor=x,
                weight_tensor=weight,
                bias_tensor=bias,
                device=self.mesh_device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=B,
                input_length=T,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                dilation=1,
                groups=1,
                return_output_dim=True,
                return_weights_and_bias=True,
            )

            self._weight_prepared, self._bias_prepared = prepared

        else:
            y = ttnn.conv1d(
                input_tensor=x,
                weight_tensor=self._weight_prepared,
                bias_tensor=self._bias_prepared,
                device=self.mesh_device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=B,
                input_length=T,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                dilation=1,
                groups=1,
            )

        return ttnn.reshape(y, (B, T, self.out_channels))

import torch
from genmo.mochi_preview.vae.models import Decoder, decode_latents


class ShapeTracker:
    def __init__(self):
        self._indent_level = 0
        self._path_stack = []

    def _get_indent(self):
        return "  " * self._indent_level

    def hook_fn(self, name):
        def hook(module, input, output):
            # Get module representation with params directly from __repr__
            module_info = str(module)
            # Get just the first line if it's multi-line
            module_info = module_info.split("\n")[0]

            input_shapes = [tuple(x.shape) for x in input if isinstance(x, torch.Tensor)]
            output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
            shape_changed = input_shapes and output_shape and input_shapes[0] != output_shape

            print(f"{self._get_indent()}[{name}] {module_info}")
            for i, shape in enumerate(input_shapes):
                print(f"{self._get_indent()}  Input {i}: {shape}")
            if shape_changed:
                print(f"{self._get_indent()}  Output: {output_shape} <!>")

        return hook

    def pre_hook_fn(self, name):
        def hook(module, input):
            self._path_stack.append(name)
            self._indent_level += 1
            print(f"{self._get_indent()}Enter: {' -> '.join(self._path_stack)}")
            return input

        return hook

    def post_hook_fn(self, name):
        def hook(module, input, output):
            print(f"{self._get_indent()}Exit: {' -> '.join(self._path_stack)}")
            self._path_stack.pop()
            self._indent_level -= 1
            return output

        return hook


def register_shape_hooks(model):
    tracker = ShapeTracker()
    for name, module in model.named_modules():
        module.register_forward_hook(tracker.hook_fn(name))
        if list(module.children()):
            module.register_forward_pre_hook(tracker.pre_hook_fn(name))
            module.register_forward_hook(tracker.post_hook_fn(name))


@torch.no_grad()
def test_decoder():
    # Initialize decoder with same parameters as in pipelines.py
    decoder = Decoder(
        out_channels=3,
        base_channels=128,
        channel_multipliers=[1, 2, 4, 6],
        temporal_expansions=[1, 2, 3],
        spatial_expansions=[2, 2, 2],
        num_res_blocks=[3, 3, 4, 6, 3],
        latent_dim=12,
        has_attention=[False, False, False, False, False],
        output_norm=False,
        nonlinearity="silu",
        output_nonlinearity="silu",
        causal=True,
    )
    print(decoder)

    # Create sample input
    batch_size = 1
    in_channels = 12
    latent_t = 28  # ((num_frames=163 - 1) // TEMPORAL_DOWNSAMPLE=6) + 1
    latent_h = 60  # height=480 // SPATIAL_DOWNSAMPLE=8
    latent_w = 106  # width=848 // SPATIAL_DOWNSAMPLE=8

    z = torch.randn(
        (batch_size, in_channels, latent_t, latent_h, latent_w),
        device="meta",
    )

    # Run inference
    decoder.eval()
    decoder = decoder.to(device="meta")

    # Usage
    register_shape_hooks(decoder)
    # Run a forward pass to trigger hooks
    output = decode_latents(decoder, z)

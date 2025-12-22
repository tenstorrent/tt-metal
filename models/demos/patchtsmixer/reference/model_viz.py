from pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from torchview import draw_graph

model = PatchTSMixerModelForForecasting(
    context_length=512,
    prediction_length=96,
    patch_length=8,
    patch_stride=8,
    num_channels=7,
    d_model=16,
    num_layers=4,
)

model_graph = draw_graph(
    model,
    input_size=(1, 512, 7),
    expand_nested=True,
    strict=False,
    hide_inner_tensors=False,
    hide_module_functions=False,
    graph_name="PatchTSMixer",
    save_graph=True,
    filename="patchtsmixer_architecture",
)

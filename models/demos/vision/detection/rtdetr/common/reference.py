# RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further decoded into scores and classes
from transformers import RTDetrForObjectDetection

# RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.

MODELS = ["PekingU/rtdetr_r18vd", "PekingU/rtdetr_r34vd", "PekingU/rtdetr_r50vd", "PekingU/rtdetr_r101vd"]
DEFAULT_MODEL = MODELS[2]

# image_processor = RTDetrImageProcessor.from_pretrained(MODEL)

# Initializing a model (with random weights) from the configuration
model = RTDetrForObjectDetection.from_pretrained(DEFAULT_MODEL)

resnet_embedder = model.model.backbone.model.encoder.stages[0]

for name, tensor in resnet_embedder.state_dict().items():
    print(name, tensor.shape)

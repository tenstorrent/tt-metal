# Swin_v2_s Demo
Demo showcasing Swin_v2_s running on Wormhole - n150 using ttnn.

## Platforms:
    WH N150

## Details:
The entry point to swin_v2_s model is TtSwinTransformer in `models/experimental/functional_swin/tt/tt_swin_transformer.py`. The model picks up certain configs and weights from Torchvision pretrained model.

## How to Run:
Use the following command to run the swin_v2_s model :
```
pytest --disable-warnings models/experimental/functional_swin/tests/test_ttnn_swin_v2_s.py
```
### Demo
Use the following command to run the demo :
```
pytest --disable-warnings models/experimental/functional_swin/demo/demo.py
```

### Model performant running with Trace+2CQ

- To be added soon

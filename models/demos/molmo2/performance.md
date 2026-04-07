### Molmo2-8B Performance Summary

This document summarizes performance changes for Molmo2-8B between branches `ssinghal/Molmo2-8B` (initial) and `ign/Molmo2_8B_new`.

- **Initial**: results from `ssinghal/Molmo2-8B`
- **Final**: results from `ign/Molmo2_8B_new`
- **Optimization %**: computed as `(Initial - Final) / Initial * 100`


### Table of Results

| Test Module | Initial (us) | Final (us) | Optimization % |
| --- | --- | --- | --- |
| models/demos/molmo2/tests/test_vision_backbone.py::test_vision_backbone_encode_only | 44,781 | 30,837 | 31.13 |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer | 55,951 | 38,747 | 30.74 |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer_feature_layers | 45,078 | 31,256 | 30.66|
| models/demos/molmo2/tests/test_image_projector.py | 2964| 2,372 | 19.97|
| models/demos/molmo2/tests/test_image_pooling.py | 465 | 512 | -10.10 |
| models/demos/molmo2/tests/test_text_block.py | 2,566 | 2,522| 1.71 |
| models/demos/molmo2/tests/test_text_mlp.py | 1,538 | 1,529 | 0.58 |
| models/demos/molmo2/tests/test_vision_full_pcc.py | 91,521 | 85,855 | 6.19 |
| models/demos/molmo2/tests/test_molmo2_model.py | 6,331 | 6,998| -10.53|

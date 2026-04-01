### Molmo2-8B Performance Summary

This document summarizes performance changes for Molmo2-8B between branches `ssinghal/Molmo2-8B` (initial) and `ign/fs/molmo2_8B`.

- **Initial**: results from `ssinghal/Molmo2-8B`
- **Final**: results from `ign/fs/molmo2_8B`
- **Optimization %**: computed as `(Initial - Final) / Initial * 100`


### Table of Results

| Test Module | Initial (us) | Final (us) | Optimization % |
| --- | --- | --- | --- |
| models/demos/molmo2/tests/test_vision_backbone.py::test_vision_backbone_encode_only | 44781 | 29326 | 34.5% |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer | 55,951 | 29,598 | 47.10014119 |
| models/demos/molmo2/tests/test_vision_transformer.py::test_vision_transformer_feature_layers | 45,078 | 29,628 | 34.2739252|
| models/demos/molmo2/tests/test_image_projector.py | 2964| 2367 | 20.1417004|
| models/demos/molmo2/tests/test_image_pooling.py | 465 | 430 | 7.52688172|
| models/demos/molmo2/tests/test_text_block.py | 2566 | 2508| 2.260327358 |
| models/demos/molmo2/tests/test_text_mlp.py | 1538 | 1511 | 1.755526658 |
| models/demos/molmo2/tests/test_vision_full_pcc.py | 91,521 | 83,123 | 9.176036101 |
| models/demos/molmo2/tests/test_molmo2_model.py | 6331 | 5848 | 7.62912652|

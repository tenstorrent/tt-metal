#  Effective Squeeze-Excitation Variety of View Network (ESE-VoVNet) model

## Details

The entry point to the Functional VoVNet model is the vovnet function located in ttnn_functional_vovnet.py. The `hf_hub:timm/ese_vovnet19b_dw.ra_in1k` version from Hugging Face is used as the reference model.


## How to Run

To run the end to end model pipeline, use the following command:

```pytest /tt-metal/models/experimental/functional_vovnet/tests/test_ttnn_functional_vovnet.py::test_vovnet```

## Components
Model card - [12997](https://github.com/tenstorrent/tt-metal/issues/12997)\
Bringup - [13000](https://github.com/tenstorrent/tt-metal/issues/13000)


## Pending issues
[#12651](https://github.com/tenstorrent/tt-metal/issues/12651) - ttnn.max_pool2d is failing with `Row size should be power of 2`

[#12648](https://github.com/tenstorrent/tt-metal/issues/12648) - ttnn.conv2d is failing with `Statically allocated l1 buffer`

[#11044](https://github.com/tenstorrent/tt-metal/issues/11044) - ttnn.MaxPool2d `output shape mismatch`

## To be done
Fixing the PCC\
Optimisation

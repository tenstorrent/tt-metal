# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT16: torch.int16,
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.FLOAT32: torch.float,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
def test_tensor_conversion_between_torch_and_tt(shape, tt_dtype, device):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)
    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.BORROWED
    else:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.OWNED

    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT8_B,
    }:
        tt_tensor = tt_tensor.to(ttl.tensor.Layout.TILE)

    if tt_dtype in {
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    if tt_dtype in {
        ttl.tensor.DataType.BFLOAT8_B,
    }:
        tt_tensor = tt_tensor.to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttl.tensor.DataType.BFLOAT8_B:
        allclose_kwargs = dict(atol=1e-2)

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip, **allclose_kwargs)
    assert passing


tt_dtype_to_np_dtype = {
    ttl.tensor.DataType.UINT16: np.int16,
    ttl.tensor.DataType.UINT32: np.int32,
    ttl.tensor.DataType.FLOAT32: np.float32,
    ttl.tensor.DataType.BFLOAT16: np.float32,
    ttl.tensor.DataType.BFLOAT8_B: np.float32,
}


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.FLOAT32,
        # ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_np(shape, tt_dtype, device):
    dtype = tt_dtype_to_np_dtype[tt_dtype]

    if dtype in {np.int16, np.int32}:
        np_tensor = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype=dtype)
    else:
        np_tensor = np.random.random(shape).astype(dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(np_tensor, tt_dtype)
    if tt_dtype in {ttl.tensor.DataType.FLOAT32, ttl.tensor.DataType.UINT32, ttl.tensor.DataType.UINT16}:
        assert tt_tensor.storage_type() == ttl.tensor.StorageType.BORROWED

    if tt_dtype in {
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.UINT16,
    }:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    np_tensor_after_round_trip = tt_tensor.to_numpy()

    assert np_tensor.dtype == np_tensor_after_round_trip.dtype
    assert np_tensor.shape == np_tensor_after_round_trip.shape

    passing = np.allclose(np_tensor, np_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
def test_serialization(tmp_path, shape, tt_dtype):
    torch.manual_seed(0)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    file_name = tmp_path / pathlib.Path("tensor.bin")
    ttl.tensor.dump_tensor(str(file_name), tt_tensor)
    torch_tensor_from_file = ttl.tensor.load_tensor(str(file_name)).to_torch()

    assert torch_tensor.dtype == torch_tensor_from_file.dtype
    assert torch_tensor.shape == torch_tensor_from_file.shape

    allclose_kwargs = {}
    if tt_dtype == ttl.tensor.DataType.BFLOAT8_B:
        allclose_kwargs = dict(atol=1e-2)

    passing = torch.allclose(torch_tensor, torch_tensor_from_file, **allclose_kwargs)
    assert passing


GOLDEN_TENSOR_STRINGS = {
    (
        ttl.tensor.DataType.UINT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT16, layout=Layout::ROW_MAJOR)\n",
    (
        ttl.tensor.DataType.UINT32,
        ttl.tensor.Layout.ROW_MAJOR,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT32, layout=Layout::ROW_MAJOR)\n",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.Layout.ROW_MAJOR,
    ): "ttnn.Tensor([[[[ 0.49626,  0.76822,  ...,  0.30510,  0.93200],\n               [ 0.17591,  0.26983,  ...,  0.20382,  0.65105],\n               ...,\n               [ 0.76926,  0.42571,  ...,  0.84923,  0.56027],\n               [ 0.44989,  0.81796,  ...,  0.82632,  0.29092]],\n\n              [[ 0.23870,  0.35561,  ...,  0.60709,  0.26819],\n               [ 0.30522,  0.16529,  ...,  0.58980,  0.36324],\n               ...,\n               [ 0.23448,  0.04438,  ...,  0.79019,  0.79197],\n               [ 0.40821,  0.77287,  ...,  0.61930,  0.06359]],\n\n              ...,\n\n              [[ 0.83083,  0.25181,  ...,  0.57106,  0.58434],\n               [ 0.36629,  0.82161,  ...,  0.59307,  0.03059],\n               ...,\n               [ 0.19764,  0.29350,  ...,  0.57648,  0.84179],\n               [ 0.63157,  0.61360,  ...,  0.61183,  0.73247]],\n\n              [[ 0.14732,  0.71010,  ...,  0.23446,  0.66704],\n               [ 0.80021,  0.18268,  ...,  0.80993,  0.10013],\n               ...,\n               [ 0.34751,  0.79996,  ...,  0.52534,  0.68817],\n               [ 0.58313,  0.48791,  ...,  0.25724,  0.24742]]],\n\n             [[[ 0.66742,  0.24011,  ...,  0.76113,  0.69809],\n               [ 0.64527,  0.37637,  ...,  0.88212,  0.59121],\n               ...,\n               [ 0.46611,  0.94733,  ...,  0.03122,  0.86672],\n               [ 0.19755,  0.84151,  ...,  0.17895,  0.65135]],\n\n              [[ 0.84791,  0.20442,  ...,  0.11282,  0.25896],\n               [ 0.79491,  0.29383,  ...,  0.44655,  0.89416],\n               ...,\n               [ 0.15174,  0.32483,  ...,  0.57135,  0.12307],\n               [ 0.12457,  0.01929,  ...,  0.79574,  0.12551]],\n\n              ...,\n\n              [[ 0.30748,  0.69975,  ...,  0.72877,  0.30830],\n               [ 0.16573,  0.45456,  ...,  0.94799,  0.36468],\n               ...,\n               [ 0.94468,  0.93938,  ...,  0.91499,  0.09071],\n               [ 0.57001,  0.48939,  ...,  0.71654,  0.78021]],\n\n              [[ 0.04604,  0.35653,  ...,  0.90001,  0.45373],\n               [ 0.09087,  0.64209,  ...,  0.97529,  0.16585],\n               ...,\n               [ 0.29423,  0.02880,  ...,  0.09598,  0.24148],\n               [ 0.29158,  0.08274,  ...,  0.43615,  0.71519]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::FLOAT32, layout=Layout::ROW_MAJOR)\n",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ): "ttnn.Tensor([[[[ 0.67188,  0.18359,  ...,  0.03516,  0.57812],\n               [ 0.44922,  0.81250,  ...,  0.79297,  0.44531],\n               ...,\n               [ 0.53516,  0.01953,  ...,  0.03906,  0.16797],\n               [ 0.73047,  0.61719,  ...,  0.51953,  0.47656]],\n\n              [[ 0.34375,  0.06641,  ...,  0.46875,  0.00781],\n               [ 0.06641,  0.55859,  ...,  0.39844,  0.57812],\n               ...,\n               [ 0.88281,  0.58594,  ...,  0.65234,  0.37891],\n               [ 0.41016,  0.68359,  ...,  0.26562,  0.36719]],\n\n              ...,\n\n              [[ 0.59766,  0.37109,  ...,  0.13281,  0.18750],\n               [ 0.46484,  0.19531,  ...,  0.72656,  0.85156],\n               ...,\n               [ 0.79688,  0.80469,  ...,  0.50781,  0.47656],\n               [ 0.55078,  0.97656,  ...,  0.10156,  0.22266]],\n\n              [[ 0.76172,  0.06641,  ...,  0.70703,  0.31250],\n               [ 0.30859,  0.05469,  ...,  0.44531,  0.13281],\n               ...,\n               [ 0.61719,  0.23047,  ...,  0.69531,  0.09375],\n               [ 0.00000,  0.97266,  ...,  0.59375,  0.18359]]],\n\n             [[[ 0.10547,  0.01953,  ...,  0.56641,  0.91406],\n               [ 0.53125,  0.08203,  ...,  0.68750,  0.66797],\n               ...,\n               [ 0.23047,  0.97656,  ...,  0.32422,  0.68359],\n               [ 0.53516,  0.92969,  ...,  0.40625,  0.89453]],\n\n              [[ 0.55859,  0.55469,  ...,  0.71875,  0.17188],\n               [ 0.40625,  0.29688,  ...,  0.19141,  0.48438],\n               ...,\n               [ 0.75391,  0.32422,  ...,  0.67188,  0.57812],\n               [ 0.85547,  0.26953,  ...,  0.43750,  0.61328]],\n\n              ...,\n\n              [[ 0.27734,  0.75000,  ...,  0.85156,  0.89453],\n               [ 0.41797,  0.36328,  ...,  0.34766,  0.62500],\n               ...,\n               [ 0.76562,  0.05078,  ...,  0.90234,  0.08203],\n               [ 0.28125,  0.57812,  ...,  0.05078,  0.94141]],\n\n              [[ 0.44141,  0.45312,  ...,  0.07031,  0.39062],\n               [ 0.41406,  0.73828,  ...,  0.61719,  0.00781],\n               ...,\n               [ 0.57812,  0.27734,  ...,  0.31250,  0.75000],\n               [ 0.07812,  0.29297,  ...,  0.43750,  0.71484]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)\n",
    (
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.ROW_MAJOR,
    ): "ttnn.Tensor([[[[ 0.50000,  0.76562,  ...,  0.30469,  0.92969],\n               [ 0.17969,  0.27344,  ...,  0.20312,  0.64844],\n               ...,\n               [ 0.76562,  0.42188,  ...,  0.85156,  0.56250],\n               [ 0.45312,  0.82031,  ...,  0.82812,  0.28906]],\n\n              [[ 0.24219,  0.35938,  ...,  0.60938,  0.26562],\n               [ 0.30469,  0.16406,  ...,  0.58594,  0.35938],\n               ...,\n               [ 0.23438,  0.04688,  ...,  0.78906,  0.78906],\n               [ 0.40625,  0.77344,  ...,  0.61719,  0.06250]],\n\n              ...,\n\n              [[ 0.82812,  0.25000,  ...,  0.57031,  0.58594],\n               [ 0.36719,  0.82031,  ...,  0.59375,  0.03125],\n               ...,\n               [ 0.19531,  0.29688,  ...,  0.57812,  0.84375],\n               [ 0.63281,  0.61719,  ...,  0.60938,  0.73438]],\n\n              [[ 0.14844,  0.71094,  ...,  0.23438,  0.66406],\n               [ 0.79688,  0.17969,  ...,  0.81250,  0.10156],\n               ...,\n               [ 0.34375,  0.79688,  ...,  0.52344,  0.68750],\n               [ 0.58594,  0.48438,  ...,  0.25781,  0.25000]]],\n\n             [[[ 0.66406,  0.24219,  ...,  0.75781,  0.69531],\n               [ 0.64844,  0.37500,  ...,  0.88281,  0.59375],\n               ...,\n               [ 0.46875,  0.94531,  ...,  0.03125,  0.86719],\n               [ 0.19531,  0.84375,  ...,  0.17969,  0.64844]],\n\n              [[ 0.85156,  0.20312,  ...,  0.10938,  0.25781],\n               [ 0.79688,  0.29688,  ...,  0.44531,  0.89062],\n               ...,\n               [ 0.14844,  0.32812,  ...,  0.57031,  0.12500],\n               [ 0.12500,  0.01562,  ...,  0.79688,  0.12500]],\n\n              ...,\n\n              [[ 0.30469,  0.70312,  ...,  0.72656,  0.30469],\n               [ 0.16406,  0.45312,  ...,  0.94531,  0.36719],\n               ...,\n               [ 0.94531,  0.93750,  ...,  0.91406,  0.09375],\n               [ 0.57031,  0.49219,  ...,  0.71875,  0.78125]],\n\n              [[ 0.04688,  0.35938,  ...,  0.89844,  0.45312],\n               [ 0.09375,  0.64062,  ...,  0.97656,  0.16406],\n               ...,\n               [ 0.29688,  0.03125,  ...,  0.09375,  0.24219],\n               [ 0.28906,  0.08594,  ...,  0.43750,  0.71875]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT8_B, layout=Layout::ROW_MAJOR)\n",
    (
        ttl.tensor.DataType.UINT16,
        ttl.tensor.Layout.TILE,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   659,   147],\n               [  183,    28,  ...,   131,   972],\n               ...,\n               [  108,   203,  ...,   225,   614],\n               [   81,   541,  ...,   389,   378]],\n\n              [[  856,   273,  ...,    26,   760],\n               [  383,   492,  ...,   974,   170],\n               ...,\n               [  758,   845,  ...,   475,   272],\n               [  696,   479,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   461,   641],\n               [  972,   301,  ...,   306,   804],\n               ...,\n               [  803,   326,  ...,   959,   655],\n               [   95,   651,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   757,   722],\n               [  120,   843,  ...,   193,   910],\n               ...,\n               [  757,   557,  ...,    66,   991],\n               [  585,   357,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   910,    43],\n               [  895,   132,  ...,   566,   607],\n               ...,\n               [  109,   737,  ...,   566,   920],\n               [  822,     2,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   666,   195],\n               [  741,   433,  ...,   517,   769],\n               ...,\n               [  356,   265,  ...,   139,   256],\n               [  928,   683,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   387,   693],\n               [  927,   148,  ...,   685,   337],\n               ...,\n               [  523,   241,  ...,   478,   221],\n               [  161,   554,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,    32,   203],\n               [  546,   115,  ...,   499,   765],\n               ...,\n               [  781,  1004,  ...,   572,   434],\n               [  587,  1005,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT16, layout=Layout::TILE)\n",
    (
        ttl.tensor.DataType.UINT32,
        ttl.tensor.Layout.TILE,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   659,   147],\n               [  183,    28,  ...,   131,   972],\n               ...,\n               [  108,   203,  ...,   225,   614],\n               [   81,   541,  ...,   389,   378]],\n\n              [[  856,   273,  ...,    26,   760],\n               [  383,   492,  ...,   974,   170],\n               ...,\n               [  758,   845,  ...,   475,   272],\n               [  696,   479,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   461,   641],\n               [  972,   301,  ...,   306,   804],\n               ...,\n               [  803,   326,  ...,   959,   655],\n               [   95,   651,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   757,   722],\n               [  120,   843,  ...,   193,   910],\n               ...,\n               [  757,   557,  ...,    66,   991],\n               [  585,   357,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   910,    43],\n               [  895,   132,  ...,   566,   607],\n               ...,\n               [  109,   737,  ...,   566,   920],\n               [  822,     2,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   666,   195],\n               [  741,   433,  ...,   517,   769],\n               ...,\n               [  356,   265,  ...,   139,   256],\n               [  928,   683,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   387,   693],\n               [  927,   148,  ...,   685,   337],\n               ...,\n               [  523,   241,  ...,   478,   221],\n               [  161,   554,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,    32,   203],\n               [  546,   115,  ...,   499,   765],\n               ...,\n               [  781,  1004,  ...,   572,   434],\n               [  587,  1005,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT32, layout=Layout::TILE)\n",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.Layout.TILE,
    ): "ttnn.Tensor([[[[ 0.49626,  0.76822,  ...,  0.81547,  0.79316],\n               [ 0.77449,  0.43689,  ...,  0.29523,  0.79669],\n               ...,\n               [ 0.71076,  0.70912,  ...,  0.64436,  0.61472],\n               [ 0.77735,  0.58957,  ...,  0.82632,  0.29092]],\n\n              [[ 0.23870,  0.35561,  ...,  0.43097,  0.35527],\n               [ 0.52108,  0.94565,  ...,  0.29352,  0.36860],\n               ...,\n               [ 0.07921,  0.57293,  ...,  0.51687,  0.37703],\n               [ 0.30046,  0.16940,  ...,  0.61930,  0.06359]],\n\n              ...,\n\n              [[ 0.83083,  0.25181,  ...,  0.94931,  0.92893],\n               [ 0.45172,  0.58718,  ...,  0.17067,  0.39629],\n               ...,\n               [ 0.61510,  0.83406,  ...,  0.78765,  0.11540],\n               [ 0.96949,  0.62071,  ...,  0.61183,  0.73247]],\n\n              [[ 0.14732,  0.71010,  ...,  0.51140,  0.65196],\n               [ 0.39442,  0.00957,  ...,  0.59083,  0.67602],\n               ...,\n               [ 0.75072,  0.73764,  ...,  0.32935,  0.44592],\n               [ 0.12882,  0.56820,  ...,  0.25724,  0.24742]]],\n\n             [[[ 0.66742,  0.24011,  ...,  0.80254,  0.04462],\n               [ 0.70165,  0.88148,  ...,  0.17447,  0.21805],\n               ...,\n               [ 0.35950,  0.40593,  ...,  0.07120,  0.77081],\n               [ 0.97368,  0.73132,  ...,  0.17895,  0.65135]],\n\n              [[ 0.84791,  0.20442,  ...,  0.06364,  0.09529],\n               [ 0.26536,  0.99807,  ...,  0.98837,  0.00548],\n               ...,\n               [ 0.38363,  0.27553,  ...,  0.01710,  0.19673],\n               [ 0.43170,  0.01451,  ...,  0.79574,  0.12551]],\n\n              ...,\n\n              [[ 0.30748,  0.69975,  ...,  0.66885,  0.60539],\n               [ 0.13134,  0.60182,  ...,  0.36747,  0.57308],\n               ...,\n               [ 0.13455,  0.11513,  ...,  0.52951,  0.01698],\n               [ 0.69117,  0.31540,  ...,  0.71654,  0.78021]],\n\n              [[ 0.04604,  0.35653,  ...,  0.11670,  0.54744],\n               [ 0.81430,  0.91083,  ...,  0.63913,  0.74077],\n               ...,\n               [ 0.20421,  0.30298,  ...,  0.79129,  0.57022],\n               [ 0.85929,  0.04492,  ...,  0.43615,  0.71519]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)\n",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
    ): "ttnn.Tensor([[[[ 0.67188,  0.18359,  ...,  0.57422,  0.57422],\n               [ 0.71484,  0.10938,  ...,  0.51172,  0.79688],\n               ...,\n               [ 0.42188,  0.79297,  ...,  0.87891,  0.39844],\n               [ 0.31641,  0.11328,  ...,  0.51953,  0.47656]],\n\n              [[ 0.34375,  0.06641,  ...,  0.10156,  0.96875],\n               [ 0.49609,  0.92188,  ...,  0.80469,  0.66406],\n               ...,\n               [ 0.96094,  0.30078,  ...,  0.85547,  0.06250],\n               [ 0.71875,  0.87109,  ...,  0.26562,  0.36719]],\n\n              ...,\n\n              [[ 0.59766,  0.37109,  ...,  0.80078,  0.50391],\n               [ 0.79688,  0.17578,  ...,  0.19531,  0.14062],\n               ...,\n               [ 0.13672,  0.27344,  ...,  0.74609,  0.55859],\n               [ 0.37109,  0.54297,  ...,  0.10156,  0.22266]],\n\n              [[ 0.76172,  0.06641,  ...,  0.95703,  0.82031],\n               [ 0.46875,  0.29297,  ...,  0.75391,  0.55469],\n               ...,\n               [ 0.95703,  0.17578,  ...,  0.25781,  0.87109],\n               [ 0.28516,  0.39453,  ...,  0.59375,  0.18359]]],\n\n             [[[ 0.10547,  0.01953,  ...,  0.55469,  0.16797],\n               [ 0.49609,  0.51562,  ...,  0.21094,  0.37109],\n               ...,\n               [ 0.42578,  0.87891,  ...,  0.21094,  0.59375],\n               [ 0.21094,  0.00781,  ...,  0.40625,  0.89453]],\n\n              [[ 0.55859,  0.55469,  ...,  0.60156,  0.76172],\n               [ 0.89453,  0.69141,  ...,  0.01953,  0.00391],\n               ...,\n               [ 0.39062,  0.03516,  ...,  0.54297,  0.00000],\n               [ 0.62500,  0.66797,  ...,  0.43750,  0.61328]],\n\n              ...,\n\n              [[ 0.27734,  0.75000,  ...,  0.51172,  0.70703],\n               [ 0.62109,  0.57812,  ...,  0.67578,  0.31641],\n               ...,\n               [ 0.04297,  0.94141,  ...,  0.86719,  0.86328],\n               [ 0.62891,  0.16406,  ...,  0.05078,  0.94141]],\n\n              [[ 0.44141,  0.45312,  ...,  0.12500,  0.79297],\n               [ 0.13281,  0.44922,  ...,  0.94922,  0.98828],\n               ...,\n               [ 0.05078,  0.92188,  ...,  0.23438,  0.69531],\n               [ 0.29297,  0.92578,  ...,  0.43750,  0.71484]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT16, layout=Layout::TILE)\n",
    (
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
    ): "ttnn.Tensor([[[[ 0.50000,  0.76562,  ...,  0.81250,  0.79688],\n               [ 0.77344,  0.43750,  ...,  0.29688,  0.79688],\n               ...,\n               [ 0.71094,  0.71094,  ...,  0.64062,  0.61719],\n               [ 0.78125,  0.58594,  ...,  0.82812,  0.28906]],\n\n              [[ 0.24219,  0.35938,  ...,  0.42969,  0.35156],\n               [ 0.52344,  0.94531,  ...,  0.29688,  0.36719],\n               ...,\n               [ 0.07812,  0.57031,  ...,  0.51562,  0.37500],\n               [ 0.29688,  0.17188,  ...,  0.61719,  0.06250]],\n\n              ...,\n\n              [[ 0.82812,  0.25000,  ...,  0.95312,  0.92969],\n               [ 0.45312,  0.58594,  ...,  0.17188,  0.39844],\n               ...,\n               [ 0.61719,  0.83594,  ...,  0.78906,  0.11719],\n               [ 0.96875,  0.61719,  ...,  0.60938,  0.73438]],\n\n              [[ 0.14844,  0.71094,  ...,  0.50781,  0.64844],\n               [ 0.39062,  0.00781,  ...,  0.59375,  0.67969],\n               ...,\n               [ 0.75000,  0.73438,  ...,  0.32812,  0.44531],\n               [ 0.12500,  0.57031,  ...,  0.25781,  0.25000]]],\n\n             [[[ 0.66406,  0.24219,  ...,  0.80469,  0.04688],\n               [ 0.70312,  0.88281,  ...,  0.17188,  0.21875],\n               ...,\n               [ 0.35938,  0.40625,  ...,  0.07031,  0.77344],\n               [ 0.97656,  0.73438,  ...,  0.17969,  0.64844]],\n\n              [[ 0.85156,  0.20312,  ...,  0.06250,  0.09375],\n               [ 0.26562,  0.99219,  ...,  0.99219,  0.00781],\n               ...,\n               [ 0.38281,  0.27344,  ...,  0.01562,  0.19531],\n               [ 0.42969,  0.01562,  ...,  0.79688,  0.12500]],\n\n              ...,\n\n              [[ 0.30469,  0.70312,  ...,  0.67188,  0.60156],\n               [ 0.13281,  0.60156,  ...,  0.36719,  0.57031],\n               ...,\n               [ 0.13281,  0.11719,  ...,  0.53125,  0.01562],\n               [ 0.68750,  0.31250,  ...,  0.71875,  0.78125]],\n\n              [[ 0.04688,  0.35938,  ...,  0.11719,  0.54688],\n               [ 0.81250,  0.91406,  ...,  0.64062,  0.74219],\n               ...,\n               [ 0.20312,  0.30469,  ...,  0.78906,  0.57031],\n               [ 0.85938,  0.04688,  ...,  0.43750,  0.71875]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT8_B, layout=Layout::TILE)\n",
}


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT16,
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize("layout", [ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.TILE])
def test_print(tt_dtype, layout):
    torch.manual_seed(0)

    assert os.environ.get("TTNN_TENSOR_PRINT_LEVEL", "SHORT") == "SHORT"

    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    shape = (2, 16, 64, 32)

    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(layout)

    tensor_as_string = str(tt_tensor)

    # To generate golden output, use the following line
    # print("\\n".join(str(tt_tensor).split("\n")))

    assert tensor_as_string == GOLDEN_TENSOR_STRINGS[(tt_dtype, layout)]

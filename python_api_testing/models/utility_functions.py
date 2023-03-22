import math

import torch
import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
)

def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag = 2.0, max_mag_fraction = 0.02):
    """
    A variant of np.isclose with logging.
    """
    absdiff = (a-b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs()+1.0) / (b.abs()+1.0) - 1.0 # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs()<rtol, reldiff2.abs()<rtol)
    max_mag_ok = (absdiff<max_mag*max_mag_fraction)

    or_abs_rel = torch.logical_or( absdiff<atol, reldiff_or )
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()
    if not or_abs_rel.reshape(-1)[debug_index]:
        print("isclose mismatch at index=", debug_index)
        print(a.reshape(-1)[debug_index])
        print(b.reshape(-1)[debug_index])
        print("reldiff1=", reldiff1.reshape(-1)[debug_index])
        print("reldiff2=", reldiff2.reshape(-1)[debug_index])
        print("absdiff=", absdiff.reshape(-1)[debug_index])
    return torch.all( or_abs_rel )


def print_diff_tt_pyt(a, b, annotation = ""):
    # first convert a pytorch tensor argument b to tt
    padded_b = pad_weight(b)
    pyt_a = tt2torch(a) # untilizes also
    return print_diff_argmax(pyt_a, padded_b, annotation)


def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    ooms = []
    for el in float_lst:
        str_el = str(el)
        if "e" in str_el:
            oom = int(str_el.split("e")[1])
        elif str_el[:2] == "0.":
            str_el = str_el.split(".")[1]

            oom = -1
            for e in str_el:
                if e != "0":
                    break
                oom -= 1
        else:
            oom = len(str_el.split(".")[0])

        ooms.append(oom)

    return ooms

def get_FR():
    # TODO(AP): a hacky workflow where we manually set force recompile counter before every kernel from python
    return ttl.device.GetForceRecompiles()

def set_FR(new_val):
    # TODO(AP): a hacky workflow where we manually set force recompile counter before every kernel from python
    ttl.device.SetForceRecompiles(new_val)
    print("Force recompiles=", get_FR())


def ttP(x, count=4, offset=0, stride=1):
    if type(x) == torch.Tensor:
        t1 = x.reshape(-1)
    else:
        host = ttl.device.GetHost()
        shp = x.shape()
        tt_out = x.to(host)
        torch_out = untilize(torch.Tensor(tt_out.data()).reshape(shp))
        t1 = torch_out.reshape(-1)
    print("Tensor vals: (", end="")
    for j in range(offset, offset+count*stride, stride):
        print(t1[j].item(), " ", end="")
    print(")")

def enable_compile_cache():
    """
    Enables the compiler caching.
    """
    ttl.device.EnableCompileCache()

def disable_compile_cache():
    """
    Disables the compiler caching.
    """
    ttl.device.DisableCompileCache()

def get_compile_cache_enabled():
    """
    Returns the current state of compile cache on/off switch.
    """
    return ttl.device.GetCompileCacheEnabled()

def enable_binary_cache():
    """
    Enables the binary loading cache.
    """
    ttl.device.EnableBinaryCache()

def disable_binary_cache():
    """
    Disables the binary loading cache.
    """
    ttl.device.DisableBinaryCache()

def get_binary_cache_enabled():
    """
    Returns the current state of binary loading cache on/off switch.
    """
    return ttl.device.GetBinaryCacheEnabled()

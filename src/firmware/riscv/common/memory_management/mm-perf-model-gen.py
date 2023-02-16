#!/usr/bin/env python3
import numpy as np
import lmfit

# NaNs are ignored.
# NaNs at the left are redundant data.
# NaNs at the right are "it takes too long to measure it and sometimes also datum is > max buffers".

ALLOCATE_INTERNAL_KEYS_DATA = [
    [290, 294, 286, 258, 456, 451, 663, 906, 1120, 1439, 2004],
    [409, 384, 408, 739, 1024, 1398, 1681, 2132, 3337, 4634, 6619],
    [452, 452, 856, 846, 1680, 2069, 2978, 4865, 7605, 11381, 18564],
    [500, 477, 977, 1434, 2350, 4001, 6098, 9630, 15552, 27157, 47520],
    [548, 514, 1067, 2131, 3193, 6079, 10622, 18065, 32921, 59808, 116476],
    [592, 592, 1738, 2928, 5466, 10436, 18417, 34855, 66675, np.nan, np.nan],
    [634, 614, 1833, 3832, 7879, 15267, 31142, 62248, np.nan, np.nan, np.nan],
]

ALLOCATE_EXTERNAL_KEYS_DATA = [
    [283, 283, 283, 277, 464, 478, 690, 892, 1125, 1376, 1839],
    [398, 410, 394, 738, 1079, 1408, 1748, 2134, 3322, 4675, 6698],
    [449, 452, 841, 865, 1676, 2083, 2994, 4982, 7537, 11322, 18606],
    [503, 489, 1002, 1433, 2393, 4012, 6187, 9794, 15686, 27398, 48125],
    [552, 543, 1098, 2182, 3321, 6321, 10945, 18508, 33404, 61018, 118019],
    [599, 593, 1759, 3011, 5593, 10638, 18803, 35465, 67529, np.nan, np.nan],
    [652, 635, 1904, 4009, 8265, 15953, 32446, 64688, np.nan, np.nan, np.nan],
]

DEALLOCATE_INTERNAL_KEYS_DATA = [
    [np.nan, 478, 460, 454, 981, 1004, 1498, 2049, 2617, 3224, 4490],
    [np.nan, np.nan, 452, 958, 1479, 2019, 2554, 3186, 5072, 7306, 10468],
    [np.nan, np.nan, 951, 945, 1995, 2552, 3739, 6425, 10320, 15803, 26793],
    [np.nan, np.nan, 951, 1434, 2506, 4281, 7008, 11811, 20703, 39705, 77040],
    [np.nan, np.nan, 920, 1918, 2988, 6147, 11638, 22587, 48157, 105645, 247922],
    [np.nan, np.nan, 1404, 2365, 4681, 9712, 20006, 47497, 118570, np.nan, np.nan],
    [np.nan, np.nan, 1335, 2892, 6506, 14720, 38021, 103546, np.nan, np.nan, np.nan],
]

DEALLOCATE_EXTERNAL_KEYS_DATA = [
    [np.nan, 483, 483, 484, 1046, 1041, 1587, 2210, 2822, 3456, 4830],
    [np.nan, np.nan, 482, 1030, 1564, 2192, 2801, 3399, 5519, 7857, 11310],
    [np.nan, np.nan, 1023, 1004, 2099, 2770, 4041, 6935, 11142, 17138, 29271],
    [np.nan, np.nan, 977, 1498, 2654, 4633, 7542, 12743, 22484, 43465, 84852],
    [np.nan, np.nan, 965, 2049, 3191, 6588, 12496, 24473, 52702, 116632, 275524],
    [np.nan, np.nan, 1475, 2582, 4982, 10442, 21611, 51973, 130894, np.nan, np.nan],
    [np.nan, np.nan, 1413, 2963, 6936, 15771, 41396, 114082, np.nan, np.nan, np.nan],
]

NUM_UNUSABLE_LEVELS = 3

def main(_):
    np.set_printoptions(
        formatter={
            'float': '{:9.1f}'.format,
            'int': '{:9}'.format,
        },
        linewidth=10000000,
    )

    all_match_initial_params = True
    print("\n=== Allocate with Internal Keys ===\n")
    all_match_initial_params &= fit_allocate_internal_keys_model()
    print("\n=== Allocate with External Keys ===\n")
    all_match_initial_params &= fit_allocate_external_keys_model()
    print("\n=== Deallocate with Internal Keys ===\n")
    all_match_initial_params &= fit_deallocate_internal_keys_model()
    print("\n=== Deallocate with External Keys ===\n")
    all_match_initial_params &= fit_deallocate_external_keys_model()

    if all_match_initial_params:
        print("\nall models closely match their initial parameters\n")
    else:
        print("\n!!! some models do not closely match their initial parameters !!!\n")


def fit_allocate_internal_keys_model():
    def allocate_model(params, *, num_ops, level_ops, **_):
        level_part = np.power(num_ops, params['num_ops_per_lvl_exp']) * np.power(level_ops, params['lvl_exp'])
        ops_part = num_ops * params['ops_coeff_1']
        return params['lvl_coeff'] * level_part + ops_part + params['intercept']

    # initial values here are the result of a good fit
    params = lmfit.Parameters()
    params.add('intercept',             value=105,  min=0,   max=300)
    params.add('lvl_coeff',             value=158,  min=0,   max=300, vary=False)
    params.add('lvl_exp',               value=0.6,  min=0.5, max=2,   vary=False)
    params.add('num_ops_per_lvl_exp',   value=1.11, min=0.5, max=2)
    params.add('ops_coeff_1',           value=24,   min=10,  max=300)

    return minimize(allocate_model, params, np.array(ALLOCATE_INTERNAL_KEYS_DATA))


def fit_allocate_external_keys_model():
    def allocate_model(params, *, num_ops, level_ops, **_):
        level_part = np.power(num_ops, params['num_ops_per_lvl_exp']) * np.power(level_ops, params['lvl_exp'])
        ops_part = num_ops * params['ops_coeff_1']
        return params['lvl_coeff'] * level_part + ops_part + params['intercept']

    # initial values here are the result of a good fit
    params = lmfit.Parameters()
    params.add('intercept',             value=114,  min=0,   max=300)
    params.add('lvl_coeff',             value=158,  min=0,   max=300, vary=False)
    params.add('lvl_exp',               value=0.6,  min=0.5, max=2,   vary=False)
    params.add('num_ops_per_lvl_exp',   value=1.13, min=0.5, max=2)
    params.add('ops_coeff_1',           value=20,   min=0,   max=300)

    return minimize(allocate_model, params, np.array(ALLOCATE_EXTERNAL_KEYS_DATA))


def fit_deallocate_internal_keys_model():
    def deallocate_model(params, *, num_ops, **_):
        linear_ops_part = params['ops_coeff_1'] * num_ops
        var_exp_ops_part = params['ops_coeff_var_exp'] * np.power(num_ops, params['ops_var_exp'])
        return linear_ops_part + var_exp_ops_part + params['intercept']

    # initial values here are the result of a good fit
    params = lmfit.Parameters()
    params.add('intercept',             value=4.9, min=-100,max=300)
    params.add('ops_coeff_1',           value=442,  min=0,   max=1000)
    params.add('ops_coeff_var_exp',     value=10.5, min=0,   max=1000)
    params.add('ops_var_exp',           value=2,    min=1.5, max=2.5)

    return minimize(deallocate_model, params, np.array(DEALLOCATE_INTERNAL_KEYS_DATA))


def fit_deallocate_external_keys_model():
    def deallocate_model(params, *, num_ops, **_):
        linear_ops_part = params['ops_coeff_1'] * num_ops
        var_exp_ops_part = params['ops_coeff_var_exp'] * np.power(num_ops, params['ops_var_exp'])
        return linear_ops_part + var_exp_ops_part + params['intercept']

    # initial values here are the result of a good fit
    params = lmfit.Parameters()
    params.add('intercept',             value=-5.3, min=-10, max=300)
    params.add('ops_coeff_1',           value=478,  min=0,   max=1000)
    params.add('ops_coeff_var_exp',     value=11.5, min=0,   max=1000)
    params.add('ops_var_exp',           value=2,    min=1.5, max=2.5)

    return minimize(deallocate_model, params, np.array(DEALLOCATE_EXTERNAL_KEYS_DATA))


def minimize(model, params, data):
    xs = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    (ops_exponent_index, level) = xs
    blocks_in_level = 2 ** (level + NUM_UNUSABLE_LEVELS)
    ops_exponent = ops_exponent_index/(data.shape[1]-1)
    num_ops = np.floor(blocks_in_level ** ops_exponent)

    # these are 'x' values passed to a model
    kws = {
        "num_ops": num_ops,
        "level_ops": level + 1,  # block maps to level N => needs to operate on N+1 levels, etc.
    }

    # relative error
    def residual(*args, **kwargs):
        return np.ravel((model(*args, **kwargs) - data)/data)

    fit_result = lmfit.minimize(residual, params, method='leastsq', kws=kws, nan_policy='omit')

    print(fit_result.params.pretty_print(fmt='f', precision=2))

    predictions = model(fit_result.params, **kws)

    print("data:\n{}\nnum ops:\n{}\npredictions:\n{}\npercent relative error:\n{}\nabsolute error:\n{}\n".format(
        data,
        num_ops,
        predictions,
        100*(predictions-data)/data,
        predictions-data,
    ))

    params_relative_change = {
        k: (final - params[k])/params[k] for k, final in fit_result.params.valuesdict().items()
    }

    percent_tolerance = 3
    params_with_significant_change = { k for k, rel_change in params_relative_change.items() if abs(rel_change) > percent_tolerance/100}
    if len(params_with_significant_change) == 0:
        print(f"all parameters are within {percent_tolerance}% of their initial value")
        return True
    else:
        print(f"!!! some parameters are NOT within {percent_tolerance}% of to their initial value !!!")
        for k in params_with_significant_change:
            print(f"{k:25}: now {fit_result.params[k].value:9.1f} was {params[k].value:9.1f} changed {params_relative_change[k]*100:+4.1f}%")
        return False



if __name__ == "__main__":
    import sys
    main(sys.argv)

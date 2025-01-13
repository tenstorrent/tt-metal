# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def get_parameter_names(parameters):
    if isinstance(parameters, dict):
        parameters = list(parameters.items())

    if len(parameters) == 0:
        return []
    else:
        first_parameter, *other_parameters = parameters
        name, _ = first_parameter
        if "," in name:
            # Mutliple parameters in one string
            names = name.split(",")
            return names + get_parameter_names(other_parameters)
        else:
            # Single parameter
            return [name] + get_parameter_names(other_parameters)


def permutations(parameters):
    if isinstance(parameters, dict):
        parameters = list(reversed(parameters.items()))

    if len(parameters) == 0:
        yield {}
    else:
        first_parameter, *other_parameters = parameters
        for permutation in permutations(other_parameters):
            name, values = first_parameter

            if "," in name:
                # Mutliple parameters in one string
                names = name.split(",")
                for value in values:
                    yield {**permutation, **dict(zip(names, value))}
            else:
                # Single parameter
                for value in values:
                    yield {**permutation, name: value}


def preprocess_parameter_value(parameter_value):
    if callable(parameter_value):
        parameter_value = parameter_value.__name__
    return parameter_value


def get_parameter_values(parameter_names, permutation):
    for parameter_name in parameter_names:
        parameter_value = preprocess_parameter_value(permutation[parameter_name])
        yield parameter_value

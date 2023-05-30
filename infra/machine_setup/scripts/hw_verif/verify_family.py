import os
from common import (
    get_smi_log_lines,
    check_same,
    check_not_empty,
    get_tt_arch_from_cmd_line,
)


def check_same(list):
    if len(list) == 0:
        return False

    for x in list:
        if x != list[0]:
            return False

    return True


def get_grayskull_family():
    log_lines = get_smi_log_lines()

    family_list = []

    for line in log_lines:
        occurs_family = line.find("Family")

        if occurs_family > -1:
            family_splitted = line.split(":")
            family_tmp = family_splitted[1]

            family_tmp_2 = family_tmp.split("\n")
            family_list.append(family_tmp_2[0].strip())

    return family_list


if __name__ == "__main__":
    family_list = get_grayskull_family()
    print(family_list)

    tt_arch = get_tt_arch_from_cmd_line()

    if tt_arch == "grayskull":
        assert (
            family_list[0] == "e150"
        ), f"At least the first card is not E150: {family_list}"
    elif tt_arch == "wormhole_b0":
        assert check_same(family_list), family_list
        assert family_list[0] in ("NEBULA_X2",)
    else:
        raise Exception(f"{tt_arch} not implemented")

import os
from common import get_smi_log_lines, check_same, check_not_empty


def get_ai_clocks():
    log_lines = get_smi_log_lines()

    freq_list  = []

    for line in log_lines:
        occurs_freq = line.find("Frequency (MHz)")

        if occurs_freq > -1:
            freq_splitted = line.split(":")
            freq_tmp = freq_splitted[1]
            freq_splitted_2 = freq_tmp.split("/")
            freq_tmp_2 = freq_splitted_2[1].split("\n")
            freq_list.append(freq_tmp_2[0].strip())

    return freq_list


if __name__ == "__main__":
    ai_clocks = get_ai_clocks()
    print(ai_clocks)

    assert check_not_empty(ai_clocks), ai_clocks
    assert check_same(ai_clocks), ai_clocks
    assert ai_clocks[0] == "1202", ai_clocks

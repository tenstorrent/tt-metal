import subprocess as sp
import copy
import os


def run_process_and_get_result(command, extra_env={}, capture_output=True):
    full_env = copy.deepcopy(os.environ)
    full_env.update(extra_env)

    result = sp.run(command, shell=True, capture_output=capture_output, env=full_env)

    return result

def get_smi_log_lines():
    tt_smi_result = run_process_and_get_result("tt-smi -s")
    output = tt_smi_result.stdout.decode()
    assert tt_smi_result.returncode == 0

    s_logname = output.split("/")

    logname = s_logname[1]
    logname = logname.split(".log")
    logname = logname[0]

    log_full_path = f"tt-smi-logs/{logname}.log"

    with open(log_full_path) as f:
        log_lines = f.readlines()

    cleanup_result = run_process_and_get_result("rm -rf tt-smi-logs")
    assert cleanup_result.returncode == 0

    return log_lines


def check_not_empty(list):
    if len(list) == 0:
        return False
    return True


def check_same(list):
    for x in list:
        if x != list[0]:
            return False

    return True

# T409_ALARM_S: raise in the main thread this many seconds after pytest starts, so a cold-JIT warm item that
# outlasts the t286 between-item deadline still ends through the test's teardown (mesh closed) before
# timeout's TERM kills python with the mesh open (job 979: killed at 560 s, then fabric-check 980 failed).
import os
import signal


def pytest_configure(config):
    secs = int(os.environ.get("T409_ALARM_S", "0"))
    if secs <= 0:
        return

    def fire(signum, frame):
        raise RuntimeError(f"[t409] alarm: {secs} s passed, stopping before the outer timeout")

    signal.signal(signal.SIGALRM, fire)
    signal.alarm(secs)

import json, os, sys, pytest

sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "timeout_method=thread", "--timeout", "1500"] + json.loads(os.environ["RA_ARGS"])))

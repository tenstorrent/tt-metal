import os, sys, pytest
ids = [l.strip() for l in open(os.environ["IDS_FILE"]) if l.strip()]
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "log_cli=true", "--log-cli-level=INFO", "-o", "timeout_method=thread"] + ids))

#!/bin/bash
set -ex

if [[ -z "$CODECHECKER_VERSION" ]]; then
  echo "::error title=Test configuration error::Missing environment variable 'CODECHECKER_VERSION'"
  exit 1
fi

cat <<EOF > ~/codechecker-server-data/docker-compose.yml
    version: '3'

    services:
      codechecker-server:
        container_name: codechecker-server
        image: "codechecker/codechecker-web:$CODECHECKER_VERSION"
        ports:
          - '8001:8001/tcp'
        networks:
          - codechecker-network
        volumes:
          - $HOME/codechecker-server-data:/workspace

    networks:
      codechecker-network:
        driver: bridge
EOF

python3 - <<- EOF
	import hashlib
	with open("$HOME/codechecker-server-data/root.user", 'w',
	          encoding='utf-8', errors='ignore') as rootf:
	    rootf.write(f"root:{hashlib.sha256(b'root:root').hexdigest()}")
EOF

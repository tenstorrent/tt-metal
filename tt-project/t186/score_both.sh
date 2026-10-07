#!/bin/bash
set -o pipefail
D=/home/smarton/fasth3/tt-metal/tt-project/t186
bash $D/score.sh s1x6; echo "s1x6 rc=$?"
bash $D/score.sh s1x5; echo "s1x5 rc=$?"

#!/bin/bash

set -ex

docker_tag="ubuntu-20.04-amd64"
docker_opts=
docker_cmd=
# Function to display help
show_help() {
    echo "Usage: $0 [-h] [-t docker tag] [-o docker launch options] -[c docker exec command]"
    echo "  -h  Show this help message."
    echo "  -t  Docker tag to run."
    echo "  -o  Docker options."
    echo "  -c  Docker exec command"
}

while getopts "t:o:c:" opt; do
    case ${opt} in
        h )
            show_help
            exit 0
            ;;
        t )
            docker_tag="$OPTARG"
            ;;
        o )
            docker_opts="$OPTARG"
            ;;
        c )
            docker_cmd="$OPTARG"
            ;;
        \? )
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "${ARCH_NAME}" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

TT_METAL_HOME=$(git rev-parse --show-toplevel)

source $TT_METAL_HOME/scripts/docker/build_docker_image.sh $docker_tag

# Allows this script to be called anywhere in the tt-metal repo
source $TT_METAL_HOME/scripts/docker/run_docker_func.sh

run_docker_common $docker_opts $docker_tag $docker_cmd

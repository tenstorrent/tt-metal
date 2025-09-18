import yaml
import os
import argparse


def multi_user_containers(num_containers, image):
    services = {}

    user_id = os.environ.get("UID") or os.getuid()
    group_id = os.environ.get("GID") or os.getgid()
    username = os.environ.get("USER") or os.getlogin()

    for i in range(0, num_containers):
        devices = []

        for n in range(0, 8):
            dev_num = (i) * 8 + n
            devices.append(f"/dev/tenstorrent/{dev_num}:/dev/tenstorrent/{dev_num}")

        service_name = f"tray-{i}"
        services[service_name] = {
            "image": image,
            "stdin_open": True,
            "tty": True,
            "container_name": service_name,
            "volumes": [
                "/home/ubuntu/actions-runner/_work/tt-metal/tt-metal:/app/tt-metal:rw",
                f"/home/ubuntu/.cache/tt-metal-cache-{i}:/app/.cache:rw",
                "/dev/hugepages-1G:/dev/hugepages-1G",
            ],
            "entrypoint": ["./.github/scripts/utils/multi-user-configure-container.sh"],
            "user": f"{user_id}:{group_id}",
            "working_dir": "/app/tt-metal",
            "environment": {
                "HOME": "/app",
                "LD_LIBRARY_PATH": "/app/tt-metal/build/lib",
                "PYTHONPATH": "/app/tt-metal",
                "TT_MESH_GRAPH_DESC_PATH": "/app/tt-metal/tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml",
                "TT_MULTI_USER_GALAXY": f"/app/tt-metal/.multi-user-galaxy-docker-files/tray-{i}.txt",
            },
            "devices": devices,
        }

        os.makedirs(f"/home/{username}/.cache/tt-metal-cache-{i}", exist_ok=True)
    return services


parser = argparse.ArgumentParser(description="Generate multi-user container YAML and hostfile.")
parser.add_argument("--num-containers", type=int, default=4, help="Number of containers to create")
parser.add_argument("--image", type=str, required=True, help="Docker image to use for containers")
args = parser.parse_args()

services = multi_user_containers(args.num_containers, args.image)

data = {"services": services}

with open("multi-user-dc.yaml", "w") as yaml_file:
    yaml.dump(data, yaml_file)

with open("hostfile", "w") as f:
    for key in services.keys():
        f.write(f"{key} slots=1\n")

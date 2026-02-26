import yaml
import os
import argparse
import shutil


def multi_user_containers(
    num_containers, image, chips_per_container, mesh_descriptor, container_prefix, tray_mapping=None
):
    services = {}

    user_id = os.environ.get("UID") or os.getuid()
    group_id = os.environ.get("GID") or os.getgid()
    username = os.environ.get("USER") or os.getlogin()

    # Validate tray_mapping if provided
    if tray_mapping:
        has_tp2_pairs = "tp2_pairs" in tray_mapping and tray_mapping["tp2_pairs"]
        has_device_mapping = "device_mapping" in tray_mapping
        if not has_tp2_pairs and not has_device_mapping:
            raise ValueError("tray_mapping must contain 'tp2_pairs' or 'device_mapping'")

    for i in range(num_containers):
        devices = []

        if tray_mapping:
            if chips_per_container == 2 and "tp2_pairs" in tray_mapping and tray_mapping["tp2_pairs"]:
                # TP2 scenario: Use Ethernet-connected pairs
                tp2_pairs = tray_mapping["tp2_pairs"]
                if i < len(tp2_pairs):
                    pair = tp2_pairs[i]
                    devices.append(f"/dev/tenstorrent/{pair[0]}:/dev/tenstorrent/{pair[0]}")
                    devices.append(f"/dev/tenstorrent/{pair[1]}:/dev/tenstorrent/{pair[1]}")
                else:
                    raise ValueError(
                        f"Not enough TP2 pairs for container {i}: have {len(tp2_pairs)}, need {num_containers}"
                    )
            else:
                # Tray scenario: Use physical topology mapping
                tray_id = i + 1  # Tray IDs are 1-indexed
                if str(tray_id) in tray_mapping["device_mapping"]:
                    device_ids = tray_mapping["device_mapping"][str(tray_id)]
                    if len(device_ids) < chips_per_container:
                        raise ValueError(
                            f"Tray {tray_id} has {len(device_ids)} devices, "
                            f"but {chips_per_container} chips per container were requested"
                        )
                    for dev_id in device_ids[:chips_per_container]:
                        devices.append(f"/dev/tenstorrent/{dev_id}:/dev/tenstorrent/{dev_id}")
                else:
                    raise ValueError(f"Tray {tray_id} not found in device_mapping")
        else:
            # Single scenario: sequential device IDs (no tray_mapping)
            for n in range(chips_per_container):
                dev_num = i * chips_per_container + n
                devices.append(f"/dev/tenstorrent/{dev_num}:/dev/tenstorrent/{dev_num}")

        service_name = f"{container_prefix}-{i}"
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
                "TT_METAL_HOME": "/app/tt-metal",
                "TTNN_RUNTIME_ARTIFACTS": f"/app/tt-metal/.ttnn_runtime_artifacts_{i}",
                "LD_LIBRARY_PATH": "/app/tt-metal/build/lib",
                "PYTHONPATH": "/app/tt-metal",
                "TT_MESH_GRAPH_DESC_PATH": f"/app/tt-metal/tt_metal/fabric/mesh_graph_descriptors/{mesh_descriptor}",
                "TT_MULTI_USER_GALAXY": f"/app/tt-metal/.multi-user-galaxy-docker-files/{service_name}.txt",
            },
            "devices": devices,
        }

        cache_path = f"/home/{username}/.cache/tt-metal-cache-{i}"
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.makedirs(cache_path)

    return services


parser = argparse.ArgumentParser(description="Generate multi-user container YAML and hostfile.")
parser.add_argument("--num-containers", type=int, default=4, help="Number of containers to create")
parser.add_argument("--image", type=str, required=True, help="Docker image to use for containers")
parser.add_argument("--chips-per-container", type=int, default=8, help="Number of chips per container (default: 8)")
parser.add_argument(
    "--mesh-descriptor",
    type=str,
    required=True,
    help="Mesh graph descriptor file name",
)
parser.add_argument(
    "--container-prefix",
    type=str,
    default="container",
    help="Prefix for container names (default: container)",
)
parser.add_argument(
    "--tray-mapping-file",
    type=str,
    help="Path to tray_to_pcie_device_mapping.yaml for physical topology-based device assignment",
)
args = parser.parse_args()

# Load tray mapping if provided
tray_mapping = None
if args.tray_mapping_file:
    with open(args.tray_mapping_file) as f:
        tray_mapping = yaml.safe_load(f)
    print(f"Loaded tray mapping from {args.tray_mapping_file}")
    print(f"Architecture: {tray_mapping.get('arch', 'unknown')}")
    print(f"Device mapping: {tray_mapping.get('device_mapping', {})}")
    print(f"TP2 pairs: {tray_mapping.get('tp2_pairs', [])}")

services = multi_user_containers(
    args.num_containers, args.image, args.chips_per_container, args.mesh_descriptor, args.container_prefix, tray_mapping
)

data = {"services": services}

with open("multi-user-dc.yaml", "w") as yaml_file:
    yaml.dump(data, yaml_file)

with open("hostfile", "w") as f:
    for key in services.keys():
        f.write(f"{key} slots=1\n")

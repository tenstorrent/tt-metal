import os
import socket


def manager_environment(plan, role, run, output, device_ids):
    local = plan[role]
    peer = "passive" if role == "source" else "source"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("KV_MANAGER_", "TT_KVM_"))
        and key
        not in {
            "KVM_ID",
            "PEERS",
            "ROLE",
            "LEADER_NAME",
            "PREFILL_TABLE",
            "DECODE_TABLE",
            "DEVICE_MAP",
            "HEALTH_PORT",
            "CONTROL_PORT",
            "ETCD_ENDPOINT",
            "MC_TCP_BIND_ADDRESS",
        }
    }
    env.update(
        {
            "KVM_ID": f"llama-{plan['run_nonce']}-{role}",
            "PEERS": f"llama-{plan['run_nonce']}-{peer}",
            "ROLE": "leader",
            "KV_MANAGER_TABLE_HOST": socket.gethostname(),
            "KV_MANAGER_PREFILL_KV_CHUNK_TABLE_PATH": str(run / "source" / "table.pb"),
            "KV_MANAGER_DECODE_KV_CHUNK_TABLE_PATH": str(run / "passive" / "table.pb"),
            "KV_MANAGER_DEVICE_MAP_PATH": str(output / "device-map.txt"),
            "KV_MANAGER_DEVICE_IDS": ",".join(map(str, device_ids)),
            "KV_MANAGER_SEAT_LOCK_DIR": "/dev/shm/tt-kv",
            "KV_MANAGER_DEVICE_IO": "dmk",
            "KV_MANAGER_DMK_ELF_PATH": plan["dmk_elf"],
            "KV_MANAGER_DISCOVERY_BACKEND": "etcd",
            "KV_MANAGER_ETCD_ENDPOINT": plan["etcd_endpoint"],
            "KV_MANAGER_ADVERTISE_HOST": local["host"],
            "KV_MANAGER_ADVERTISE_PORT": str(local["manager_control_port"]),
            "KV_MANAGER_CONTROL_MSG_ENDPOINT": f"tcp://0.0.0.0:{local['manager_control_port']}",
            "KV_MANAGER_TRANSPORT_KIND": "zmq",
            "KV_MANAGER_TRANSPORT_ENDPOINT": f"tcp://127.0.0.1:{local['manager_port']}",
            "KV_MANAGER_HTTP_PORT": str(local["health_port"]),
            "KV_MANAGER_TRANSFER_ENGINE_PROTOCOL": "tcp",
            "KV_MANAGER_TRANSFER_ENGINE_METADATA": plan["etcd_endpoint"].replace("http://", "etcd://", 1),
            "KV_MANAGER_STAGING_BUFFER_BYTES": "268435456",
            "KV_MANAGER_SLAB_COUNT": "32",
            "KV_MANAGER_TABLE_LOAD_MAX_RETRIES": "12",
            "KV_MANAGER_MIGRATION_TIMEOUT_MS": "120000",
            "LD_LIBRARY_PATH": plan["ld_library_path"],
            "TT_LOG_LEVEL": "info",
        }
    )
    return env

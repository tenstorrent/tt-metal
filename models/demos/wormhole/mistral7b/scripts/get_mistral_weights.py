# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import tarfile
import os
import urllib.request
import sys
import time
import argparse


# Show progress bar when downloading files
def show_progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


# Update path subfolder member structure
def members(tf):
    l = len("mistral-7B-v0.1/")
    for member in tf.getmembers():
        if member.path.startswith("mistral-7B-v0.1/"):
            member.path = member.path[l:]
            yield member


def download_and_untar_weights(consolidated_weights_path, instruct):
    # Check if weights exist in the specified folder. If not download and untar them.
    if not os.path.isfile(consolidated_weights_path + "/consolidated.00.pth"):
        if instruct:
            url = "https://models.mistralcdn.com/mistral-7b-v0-2/Mistral-7B-v0.2-Instruct.tar"
            downloaded_tarfile = consolidated_weights_path + "/Mistral-7B-v0.2-Instruct.tar"
        else:
            url = "https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar"
            downloaded_tarfile = consolidated_weights_path + "/mistral-7B-v0.1.tar"
        print(
            f"consolidated.00.pth not found inside folder {consolidated_weights_path}. \nDownloading weights tarfile from {url} to folder {downloaded_tarfile}"
        )

        # Download the file. Avoid download the tar file if it already exists
        if not os.path.isfile(downloaded_tarfile):
            # headers required to download the file, otherwise no permission
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0"), ("Connection", "keep-alive")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, downloaded_tarfile, show_progress)
        print(f"tarfile downloaded: {downloaded_tarfile}")

        # Untar the file
        print(f"Extracting {downloaded_tarfile} to {consolidated_weights_path}...")
        with tarfile.open(downloaded_tarfile) as tar:
            if instruct:  # the Instruct tar file does not contain any subfolders.
                tar.extractall(consolidated_weights_path)
            else:  # The general weights tarfile contains a subfolder `mistral-7B-v0.1` that needs to be removed with the members function
                tar.extractall(path=consolidated_weights_path, members=members(tar))

        # Remove the downloaded tar file
        print(f"Removing downloaded tar file {downloaded_tarfile}...")
        os.remove(downloaded_tarfile)

        # Final assert to check if the weights are present
        assert os.path.isfile(
            consolidated_weights_path + "/consolidated.00.pth"
        ), f"Weights not found inside {consolidated_weights_path} after download and extraction."
        print(f"Weights downloaded and extracted successfully to {consolidated_weights_path}!")
    else:
        print(f"{consolidated_weights_path}/consolidated.00.pth file already present")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, help="Path to store the consolidated weights folder", required=True)
    parser.add_argument(
        "--instruct_weights", action="store_true", help="Choose instruct weights to download instead of general weights"
    )

    args = parser.parse_args()

    download_and_untar_weights(consolidated_weights_path=args.weights_path, instruct=args.instruct_weights)

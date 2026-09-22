"""Browser checks against a running server. Requires optional playwright."""

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 1512, "height": 1200}, reduced_motion="reduce"
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto("http://127.0.0.1:8765")
        assert response.status == 200
        page.wait_for_function(
            "window.topkLab && window.topkLab.trace.events.length === 329"
        )
        assert page.locator(".memory-datum").count() == 256
        assert page.locator(".register-slot.empty").count() == 128
        assert page.locator(".memory-datum.row-focus").evaluate_all(
            '(xs) => xs.every(x => x.dataset.destRow === "0")'
        )
        assert (
            page.request.get("http://127.0.0.1:8765/api/source")
            .json()["kernel"]["path"]
            .endswith("ckernel_sfpu_topk.h")
        )
        page.screenshot(path="/tmp/topk-desktop-intro.png", full_page=True)

        # Keyboard progression, actual register load, and source viewer.
        page.locator("body").click(position={"x": 1000, "y": 170})
        page.keyboard.press("ArrowRight")
        assert page.locator(".memory-datum.row-focus").evaluate_all(
            '(xs) => xs.every(x => x.dataset.destCol === "0")'
        )
        for _ in range(3):
            page.keyboard.press("ArrowRight")
        assert page.evaluate("window.topkLab.event.kind") == "load"
        assert page.locator(".reg-datum").count() == 128
        assert page.locator(".memory-datum.addressed").count() == 32
        assert page.evaluate("window.topkLab.registers[0].map(v => v.value)") == [
            row * 128 + column for column in range(4) for row in range(8)
        ]
        page.screenshot(path="/tmp/topk-dest-load.png", full_page=True)
        # Inspect each individual load, then change column parity and face.
        page.locator('#load-mapping [data-register="1"]').click()
        assert page.evaluate("window.topkLab.footprint.rows") == [4, 5, 6, 7]
        assert page.locator(".addressed").count() == 32
        page.locator("#focus-row").select_option("1")
        assert page.evaluate("window.topkLab.footprint.columns") == list(
            range(1, 16, 2)
        )
        assert page.evaluate("window.topkLab.footprint.effectiveAddress") == 6
        page.locator("#focus-row").select_option("2")
        assert page.locator(".reg-datum.row-focus").evaluate_all(
            '(xs) => xs.every(x => ["1", "9", "17", "25"].includes(x.parentElement.dataset.lane))'
        )
        page.locator("#focus-row").select_option("16")
        assert page.evaluate("window.topkLab.selectedFace") == 1
        page.locator('[data-face="1:3"]').click()
        assert not page.locator("#follow-load").is_checked()
        assert page.evaluate("window.topkLab.selectedTile") == 1
        page.locator('#load-mapping [data-register="0"]').click()
        page.locator("#focus-row").select_option("0")
        assert page.locator("#follow-load").is_checked()
        page.locator("#source-button").click()
        assert page.locator("dialog").is_visible()
        assert "TTI_SFPLOAD" in page.locator(".dialog-source").inner_text()
        page.keyboard.press("Escape")
        page.locator(".reg-datum").first.click()
        assert "Input[0," in page.locator("#value-inspector").inner_text()
        page.locator("#next").click()
        assert page.evaluate("window.topkLab.event.kind") == "transpose"
        page.locator("#next").click()
        assert page.evaluate("window.topkLab.event.kind") == "swap"
        assert page.locator(".pair").count() == 4
        page.screenshot(path="/tmp/topk-desktop-swap.png", full_page=True)

        # Both trace modes, precise backwards stepping, autoplay and scrub.
        page.locator("#full").click()
        assert page.locator("#timeline").get_attribute("max") == "328"
        before = page.evaluate("window.topkLab.index")
        page.locator("#previous").click()
        assert page.evaluate("window.topkLab.index") == before - 1
        page.locator("#speed").select_option("4")
        page.locator("#play").click()
        page.wait_for_function(f"window.topkLab.index > {before}")
        page.locator("#play").click()
        stopped = page.evaluate("window.topkLab.index")
        page.wait_for_timeout(800)
        assert page.evaluate("window.topkLab.index") == stopped
        page.locator("#timeline").fill("240")
        assert page.evaluate("window.topkLab.index") == 240
        page.locator("#guided").click()
        assert page.evaluate("window.topkLab.mode") == "guided"

        # Result, export, alternate signed data and all reference dialogs.
        page.locator('#chapter-rail [data-chapter="4"]').click()
        assert page.locator("#result-section").is_visible()
        assert page.locator(".result-datum").count() == 32
        assert page.evaluate("window.topkLab.trace.output.map(x => x.value)") == list(
            range(63, 31, -1)
        )
        assert page.evaluate("window.topkLab.selectedTile") == 0
        with page.expect_download() as download:
            page.locator("#export").click()
        assert download.value.suggested_filename == "blackhole-topk-trace.json"
        page.locator("#preset").select_option("negative")
        assert page.evaluate("window.topkLab.index") == 0
        page.locator('#chapter-rail [data-chapter="4"]').click()
        assert page.evaluate("window.topkLab.trace.output.map(x => x.value)") == list(
            range(24, -8, -1)
        )
        for kind in ["lanes", "map", "scope"]:
            page.locator(f'.sidebar [data-dialog="{kind}"]').click()
            assert page.locator("dialog").is_visible()
            page.keyboard.press("Escape")
        page.locator("#original-button").click()
        assert page.locator(".original-datum").count() == 2048
        page.keyboard.press("Escape")

        # Following the bottom half changes face 2 into face 1 on unpack.
        page.locator("#focus-row").select_option("16")
        page.locator("#reset").click()
        assert page.evaluate("window.topkLab.selectedFace") == 2
        page.locator("#next").click()
        assert page.evaluate("window.topkLab.selectedFace") == 1
        assert page.locator(".memory-datum.row-focus").count() == 16
        page.locator("#preset").select_option("coordinates")
        assert page.evaluate("window.topkLab.selectedFace") == 2
        page.locator("#focus-row").select_option("0")

        # Responsive geometry: no horizontal overflow at desktop/tablet/phone.
        for width in [1512, 1280, 1024, 768, 390]:
            page.set_viewport_size({"width": width, "height": 1100})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ), f"Overflow at {width}px"
        page.locator("#reset").click()
        page.screenshot(path="/tmp/topk-mobile.png", full_page=True)
        assert not errors, errors
        # Exercise the animation path with real motion, including the SVG paths.
        page.emulate_media(reduced_motion="no-preference")
        page.set_viewport_size({"width": 1512, "height": 1100})
        page.reload()
        page.wait_for_function("window.topkLab")
        page.evaluate("window.topkLab.goTo(3)")
        page.locator("#next").click()
        assert page.locator("#movement-overlay path").count() == 16
        assert page.evaluate("document.getAnimations().length > 0")
        page.wait_for_timeout(850)
        page.locator("#next").click()
        assert page.locator("#movement-overlay path").count() == 12
        page.wait_for_timeout(850)
        page.locator("#next").click()
        assert page.locator("#movement-overlay path").count() > 0
        page.screenshot(path="/tmp/topk-animation.png", full_page=True)
        assert not errors, errors
        browser.close()
        print(
            "PASS: full DEST face, 128 lanes, individual load footprints, odd/even columns, face transpose, browser controls, source, result/export and 5 viewport sizes."
        )
        print(
            "Screenshots: /tmp/topk-desktop-intro.png, /tmp/topk-desktop-swap.png, /tmp/topk-mobile.png"
        )


if __name__ == "__main__":
    main()

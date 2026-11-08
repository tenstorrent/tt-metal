# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_DRAFT_VOCAB_TOPK with its full-head fallback (QWEN36_DRAFT_VOCAB_FALLBACK, tt/draft_vocab.py) on the chained greedy
engine (QWEN36_SPEC_FUSED=1, QWEN36_SPEC_CHAIN=1, K=3, B=8), device tests. Run with the r3 flags (QWEN36_SPEC_ARGMAX_FUSED=1
is required for the pruned head).

  * test_fallback_switch      the drafter head alternates every step (the guard's decision monkeypatched) while the live rows
                              drop 8 -> 4 -> 2 -> 1: the pruned (D) and the full-head (F) drafter trace of every bucket replay,
                              every chain step is audited, no program compiles after capture, and the streams equal plain
                              greedy up to near ties. QWEN36_DRAFT_VOCAB_TOPK defaults to 131072 here.
  * test_multilingual_accept  acceptance per language: three B=8 batches of natural text (the review's zh zh fr fr ru es en en
                              batch; the covered languages en code fr es de pt it ru; the uncovered zh ja ko ar vi hi tr pl),
                              50 greedy steps each from the prompt as is (no chat template). One JSON line per batch: committed
                              tokens per row-iteration per language, the accept histogram, the drafter head per step and a hash
                              of each row's tokens. Run once without QWEN36_DRAFT_VOCAB_TOPK and once with it and compare the
                              lines (same hashes = same greedy text). With the fallback on, a batch holding an uncovered row
                              must draft with the full head (>= 90% of its steps).

    MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 QWEN36_SPEC_SDPA_PACKED=1 QWEN36_SPEC_ARGMAX_FUSED=1 QWEN36_KV_WRITE_MULTIROW=1 \\
        QWEN36_MTP_KV_ONLY=1 [QWEN36_DRAFT_VOCAB_TOPK=131072] pytest models/demos/blackhole/qwen36/tests/test_draft_vocab_engine.py -s
"""

import hashlib
import json
import os

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import (
    BIG,
    ChainDriver,
    _env,
    _greedy_refs,
    _marks,
    _report,
    _setup,
    _streams,
)
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine import _admit_all

ROWS_REVIEW = [  # the review's batch (draft-vocab-prune review, 2026-10-29)
    ("zh", "中国是茶的故乡，饮茶的历史可以追溯到几千年以前。相传神农尝百草，日遇七十二毒，得荼而解之，这里的“荼”就是后来的茶。到了唐代，饮茶之风盛行于全国，陆羽写成了世界上第一部茶学专著《茶经》，系统地总结了茶的起源、种植、采摘、制作和品饮方法。宋代的人喜欢点茶和斗茶，把茶末放在茶盏里，用沸水冲点，再用茶筅击拂，使茶汤表面泛起细腻的白沫。明清以后，散茶逐渐取代了团茶，泡茶的方法也变得更加简单，人们更加注重茶叶本身的色、香、味、形。如今，中国的茶叶品种繁多，大致可以分为绿茶、红茶、乌龙茶、白茶、黄茶和黑茶六大类。"),
    ("zh", "长城是中国古代最伟大的军事防御工程之一。早在春秋战国时期，各诸侯国为了防御邻国的进攻，就开始在边境修筑城墙。秦始皇统一六国以后，把原来秦、赵、燕三国北方的城墙连接起来，并加以扩建，形成了西起临洮、东至辽东的万里长城。汉代又继续修筑长城，把它延伸到了今天的新疆境内。现在人们看到的长城，大部分是明代修建的，它东起鸭绿江畔，西至甘肃的嘉峪关，全长八千八百多公里。长城沿线设有许多关隘和烽火台，一旦发现敌情，守军就会白天放烟、夜间举火，把消息迅速传递出去。"),
    ("fr", "Paris est l'une des villes les plus anciennes d'Europe. Au IIIe siècle avant notre ère, une tribu gauloise, les Parisii, s'installa sur une île de la Seine, l'actuelle île de la Cité. Après la conquête romaine, la ville prit le nom de Lutèce et se développa sur la rive gauche du fleuve, où l'on construisit des thermes, un forum et des arènes. Au Moyen Âge, Paris devint la capitale du royaume de France et l'un des grands centres intellectuels du continent : l'université, fondée au début du XIIIe siècle, attirait des étudiants venus de toute la chrétienté. Sous le Second Empire, le préfet Haussmann transforma profondément la ville : il fit percer de larges boulevards, aménager des parcs et construire des égouts modernes."),
    ("fr", "La cuisine française est réputée dans le monde entier pour la richesse de ses traditions régionales. En Bourgogne, on prépare le bœuf bourguignon, un ragoût mijoté pendant plusieurs heures dans du vin rouge avec des lardons, des oignons et des champignons. En Provence, la ratatouille associe les légumes du soleil : tomates, aubergines, courgettes et poivrons, parfumés à l'huile d'olive et aux herbes. La Bretagne est célèbre pour ses crêpes et ses galettes de sarrasin, tandis que l'Alsace propose la choucroute garnie et la tarte flambée."),
    ("ru", "Москва — столица России и один из крупнейших городов Европы. Впервые она упоминается в летописи в 1147 году, когда князь Юрий Долгорукий пригласил своего союзника на пир в небольшое поселение на берегу Москвы-реки. В XIV–XV веках Москва стала центром объединения русских земель, а её правители приняли титул великих князей, а затем и царей. Кремль, окружённый мощными краснокирпичными стенами, был построен итальянскими мастерами в конце XV века. В 1812 году, во время нашествия Наполеона, большая часть города сгорела, однако уже через несколько десятилетий Москва была восстановлена."),
    ("es", "Madrid es la capital de España y la ciudad más poblada del país. Su origen se remonta al siglo IX, cuando el emir Mohamed I mandó construir una fortaleza junto al río Manzanares para defender los territorios de Al-Ándalus. En 1561, el rey Felipe II trasladó la corte a Madrid, que desde entonces se convirtió en el centro político del reino. Durante los siglos XVII y XVIII se construyeron algunos de sus monumentos más conocidos, como la Plaza Mayor, el Palacio Real y la Puerta de Alcalá."),
    ("en", "London is one of the oldest cities in Europe. Around AD 47, the Romans founded a settlement called Londinium on the north bank of the River Thames, where they built a bridge, a forum and a basilica. After the Romans left Britain in the early fifth century, the walled city was largely abandoned, and the Anglo-Saxons established a new trading town called Lundenwic a little further west. In the Middle Ages, London grew into the largest city and the commercial heart of England. The Great Fire of 1666 destroyed most of the medieval city, and Sir Christopher Wren rebuilt dozens of churches, including St Paul's Cathedral."),
    ("en", "Italian cooking is famous around the world for the variety of its regional traditions. In Emilia-Romagna, cooks prepare fresh egg pasta such as tagliatelle and tortellini, often served with a slow-cooked meat sauce. In Naples, pizza is baked for barely a minute in a wood-fired oven so hot that the crust blisters while the centre stays soft. Sicily is known for arancini, fried rice balls filled with ragù or cheese, and for cannoli stuffed with sweetened ricotta. In the north, risotto and polenta are more common than pasta."),
]
_ML = {  # one passage per language on the same topic (the review's tokenizer probe), plus Polish
    "zh": "机器学习是人工智能的一个重要分支。它通过分析大量数据来发现规律，并利用这些规律对未知数据进行预测。近年来，深度学习在图像识别、语音识别和自然语言处理等领域取得了显著的进展。不过，模型的可解释性和数据隐私问题仍然需要进一步研究。请用简单的语言解释一下什么是神经网络，并举一个日常生活中的例子。",
    "ja": "機械学習は人工知能の重要な分野の一つです。大量のデータを分析して規則性を見つけ、その規則を使って未知のデータを予測します。最近では、深層学習が画像認識や音声認識、自然言語処理などの分野で大きな成果を上げています。ニューラルネットワークとは何か、簡単に説明してください。",
    "ko": "머신러닝은 인공지능의 중요한 분야 중 하나입니다. 대량의 데이터를 분석하여 규칙을 발견하고, 그 규칙을 이용해 새로운 데이터를 예측합니다. 최근 딥러닝은 이미지 인식, 음성 인식, 자연어 처리 분야에서 큰 성과를 거두었습니다. 신경망이 무엇인지 쉽게 설명해 주세요.",
    "ru": "Машинное обучение является одним из важнейших направлений искусственного интеллекта. Оно позволяет находить закономерности в больших объёмах данных и использовать их для прогнозирования. В последние годы глубокое обучение добилось значительных успехов в распознавании изображений и обработке естественного языка. Объясните, пожалуйста, простыми словами, что такое нейронная сеть.",
    "ar": "التعلم الآلي هو أحد الفروع المهمة للذكاء الاصطناعي. يقوم بتحليل كميات كبيرة من البيانات لاكتشاف الأنماط واستخدامها للتنبؤ بالبيانات الجديدة. في السنوات الأخيرة حقق التعلم العميق تقدما كبيرا في التعرف على الصور ومعالجة اللغة الطبيعية.",
    "hi": "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक महत्वपूर्ण शाखा है। यह बड़ी मात्रा में डेटा का विश्लेषण करके पैटर्न खोजती है और उनका उपयोग नए डेटा के बारे में भविष्यवाणी करने के लिए करती है। कृपया सरल शब्दों में समझाइए कि न्यूरल नेटवर्क क्या है।",
    "fr": "L'apprentissage automatique est une branche importante de l'intelligence artificielle. Il permet de découvrir des régularités dans de grandes quantités de données et de les utiliser pour faire des prédictions. Ces dernières années, l'apprentissage profond a réalisé des progrès considérables dans la reconnaissance d'images et le traitement du langage naturel. Pouvez-vous expliquer simplement ce qu'est un réseau de neurones ?",
    "es": "El aprendizaje automático es una rama importante de la inteligencia artificial. Permite descubrir patrones en grandes cantidades de datos y utilizarlos para hacer predicciones. En los últimos años, el aprendizaje profundo ha logrado avances notables en el reconocimiento de imágenes y el procesamiento del lenguaje natural. ¿Podrías explicar de forma sencilla qué es una red neuronal?",
    "pt": "O aprendizado de máquina é um ramo importante da inteligência artificial. Ele permite descobrir padrões em grandes quantidades de dados e usá-los para fazer previsões. Nos últimos anos, o aprendizado profundo alcançou avanços notáveis no reconhecimento de imagens e no processamento de linguagem natural. Você pode explicar de forma simples o que é uma rede neural?",
    "de": "Maschinelles Lernen ist ein wichtiger Teilbereich der künstlichen Intelligenz. Es ermöglicht, Muster in großen Datenmengen zu erkennen und diese für Vorhersagen zu nutzen. In den letzten Jahren hat das Deep Learning bemerkenswerte Fortschritte bei der Bilderkennung und der Verarbeitung natürlicher Sprache erzielt. Können Sie einfach erklären, was ein neuronales Netz ist?",
    "it": "L'apprendimento automatico è un ramo importante dell'intelligenza artificiale. Permette di scoprire schemi in grandi quantità di dati e di usarli per fare previsioni. Negli ultimi anni, l'apprendimento profondo ha ottenuto progressi notevoli nel riconoscimento delle immagini e nell'elaborazione del linguaggio naturale.",
    "vi": "Học máy là một nhánh quan trọng của trí tuệ nhân tạo. Nó cho phép phát hiện các quy luật trong lượng lớn dữ liệu và sử dụng chúng để đưa ra dự đoán. Trong những năm gần đây, học sâu đã đạt được những tiến bộ đáng kể trong nhận dạng hình ảnh và xử lý ngôn ngữ tự nhiên.",
    "tr": "Makine öğrenimi, yapay zekânın önemli bir dalıdır. Büyük miktarda veride örüntüler bulmayı ve bunları tahmin yapmak için kullanmayı sağlar. Son yıllarda derin öğrenme, görüntü tanıma ve doğal dil işleme alanlarında önemli ilerlemeler kaydetmiştir.",
    "pl": "Uczenie maszynowe jest jedną z najważniejszych dziedzin sztucznej inteligencji. Pozwala odkrywać wzorce w dużych zbiorach danych i wykorzystywać je do przewidywania. W ostatnich latach głębokie uczenie osiągnęło znaczące postępy w rozpoznawaniu obrazów i przetwarzaniu języka naturalnego. Wyjaśnij, proszę, prostymi słowami, czym jest sieć neuronowa.",
    "code": "import asyncio\nfrom dataclasses import dataclass, field\n\n@dataclass\nclass Job:\n    id: int\n    deps: list[int] = field(default_factory=list)\n\nasync def run(jobs: dict[int, Job]) -> list[int]:\n    order, seen = [], set()\n    def visit(j):\n        if j in seen:\n            return\n        seen.add(j)\n        for d in jobs[j].deps:\n            visit(d)\n        order.append(j)\n",
}
BATCHES = {
    "review": ROWS_REVIEW,
    "covered": [ROWS_REVIEW[6]] + [(k, _ML[k]) for k in ("code", "fr", "es", "de", "pt", "it", "ru")],
    "uncovered": [(k, _ML[k]) for k in ("zh", "ja", "ko", "ar", "vi", "hi", "tr", "pl")],
}
UNCOVERED = {"zh", "ja", "ko", "ar", "vi", "hi", "tr", "pl"}
STEPS = 50


def _flag_env(monkeypatch):
    monkeypatch.setenv("QWEN36_DRAFT_VOCAB_TOPK", os.environ.get("QWEN36_DRAFT_VOCAB_TOPK") or "131072")
    monkeypatch.setenv("QWEN36_DRAFT_VOCAB_FALLBACK", "1")


@_marks
def test_fallback_switch(mesh_device, monkeypatch):
    _env(monkeypatch)
    _flag_env(monkeypatch)
    model, eng, tokenizer, prompts = _setup(mesh_device)
    assert model.mtp._vocab_prune is not None, "no pruned drafter head (QWEN36_SPEC_ARGMAX_FUSED=1 is required)"
    assert eng._dv_guard is not None and eng._dr_traces_full, "no full-head drafter traces"
    assert set(eng._dr_traces_full) == {eng.B, *eng._dr_bks}, (sorted(eng._dr_traces_full), eng._dr_bks)
    state = {"n": 0}

    def alternate(rows):  # the guard's decision: the full head every other step
        state["n"] += 1
        return state["n"] % 2 == 0

    monkeypatch.setattr(eng._dv_guard, "any_off", alternate)
    replayed = set()
    orig = eng._dr_trace_for

    def spy(b):
        tid = orig(b)
        replayed.add(("F" if eng._dv_full_now else "D", eng.B if b is None else int(b), tid))
        return tid

    monkeypatch.setattr(eng, "_dr_trace_for", spy)
    n_prog = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        drv = ChainDriver(eng, seed=7, sampled=lambda s: False)
        _admit_all(drv, prompts, 8, BIG)
        for live_after, steps in ((8, 60), (4, 30), (2, 20), (1, 20)):
            for r in [r for r in drv.live() if r >= live_after]:
                drv._finish(r)
            for _ in range(steps):
                drv.step()
        cb, fb = _report(eng, drv, "vocab_fallback_switch")
        assert eng.chain_steps >= 60 and eng.chain_audits == eng.chain_steps
        assert mesh_device.num_program_cache_entries() == n_prog, "a program compiled after capture"
        streams = _streams(drv)
        for r in drv.live():
            drv._finish(r)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    heads = {(h, b) for h, b, _ in replayed}
    logger.info(f"[vocab:switch] head steps {eng.dv_head_steps}; drafter traces replayed {sorted(heads)}")
    assert eng.dv_head_steps["full"] >= 60 and eng.dv_head_steps["pruned"] >= 60, eng.dv_head_steps
    for b in {eng.B, *eng._dr_bks} & {8, 4, 2, 1}:
        assert ("D", b) in heads and ("F", b) in heads, f"bucket {b}: replayed {sorted(heads)}"
    full_ids = {t for h, _, t in replayed if h == "F"}
    assert full_ids <= set(eng._dr_traces_full.values()) and not (full_ids & {t for h, _, t in replayed if h == "D"})
    model.free_kv_caches()
    model = eng = drv = None
    _greedy_refs(mesh_device, tokenizer, prompts, streams, "vocab fallback switch")


@_marks
def test_multilingual_accept(mesh_device, monkeypatch):
    _env(monkeypatch)
    model, eng, tokenizer, _ = _setup(mesh_device)
    guard = eng._dv_guard
    summary, rid = {}, 0
    for name, rows in BATCHES.items():
        h0 = dict(getattr(eng, "dv_head_steps", {"pruned": 0, "full": 0}))
        drv = ChainDriver(eng, seed=3, sampled=lambda s: False)
        for u, (_, text) in enumerate(rows):
            drv.admit(u, rid + u, tokenizer(text, add_special_tokens=False)["input_ids"], max_new=BIG)
        for _ in range(STEPS):
            drv.step()
        per_row, by_lang = [], {}
        for u, (lang, _) in enumerate(rows):
            ms = [s["m"][u] for s in drv.stats if u in s["m"]]
            out = list(drv.reqs[u]["out"])
            cpi = (sum(ms) + len(ms)) / max(1, len(ms))
            per_row.append(
                dict(
                    lang=lang,
                    iters=len(ms),
                    committed_per_iter=round(cpi, 3),
                    hist=[ms.count(j) for j in range(tc.K + 1)],
                    tokens=len(out),
                    md5=hashlib.md5(str(out).encode()).hexdigest()[:12],
                )
            )
            by_lang.setdefault(lang, []).append(cpi)
        for u in list(drv.live()):
            drv._finish(u)
        rid += len(rows)
        h1 = dict(getattr(eng, "dv_head_steps", {"pruned": 0, "full": 0}))
        heads = {k: h1[k] - h0.get(k, 0) for k in h1}
        summary[name] = dict(
            committed_per_iter={k: round(sum(v) / len(v), 3) for k, v in by_lang.items()},
            heads=heads,
            rows=per_row,
        )
        logger.info(f"[vocab:ml] {name} " + json.dumps(summary[name], ensure_ascii=False))
        if guard is not None and any(lang in UNCOVERED for lang, _ in rows):
            assert heads["full"] >= 0.9 * STEPS, f"{name}: an uncovered row drafted with the pruned head ({heads})"
    flag = os.environ.get("QWEN36_DRAFT_VOCAB_TOPK", "0")
    fb = "on" if guard is not None else "off"
    logger.info(
        f"[vocab:ml-summary] topk={flag} fallback={fb} "
        + json.dumps({k: v["committed_per_iter"] for k, v in summary.items()}, ensure_ascii=False)
    )
